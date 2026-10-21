"""
Canonical wire encoding for the handshake and ratchet messages.

Nothing in ``src/crypto`` can be sent anywhere: :class:`~src.crypto.x3dh.HandshakeInit`
and :class:`~src.crypto.x3dh.PreKeyBundle` are dataclasses with no byte
representation. This module is that representation, and it is the boundary where
untrusted bytes enter the process, so the rules are strict:

- every decoder is total: on hostile input it raises :class:`WireError` and
  nothing else. No ``KeyError``, no ``struct.error``, no unbounded loop, no
  allocation proportional to an attacker-declared length.
- lengths are varints in canonical form. A non-minimal encoding such as
  ``0x80 0x00`` is rejected, because accepting two encodings of the same frame
  lets an attacker produce two distinct byte strings for one message, which
  breaks any future signature over the encoding.
- the frame length is checked against :data:`MAX_FRAME_BYTES` *before* any
  buffer is built, so a declared length of 2**40 costs nothing.
- decoders consume the whole input. Trailing bytes are an error, not padding.
- fields whose value is derived rather than transmitted are re-derived here, so
  they cannot be tampered with. ``HandshakeInit.identity_key_x25519`` is the
  case in point: the initiator signs a transcript containing it, the receiver
  derives it from ``identity_key``, and it never appears on the wire.

Frame layout::

    +---------+---------+------------------+-------------------+
    | version | kind    | payload_len      | payload           |
    | 1 byte  | 1 byte  | varint (1..9 B)  | payload_len bytes |
    +---------+---------+------------------+-------------------+

There is no length prefix on ``version``/``kind``: a frame is at least three
bytes, and the first byte alone selects the format version.
"""

from dataclasses import dataclass
import os
from typing import Optional, Tuple

from ..crypto.double_ratchet import PROTOCOL_VERSION
from ..crypto.x3dh import (
    HandshakeInit,
    KeyConfirmation,
    KeyConfirmationFailed,
    PreKeyBundle,
)
from ..crypto.key_management import identity_public_x25519
from ..crypto.pq import PQError, load_public_key as load_pq_public_key
from nacl.exceptions import BadSignatureError
from nacl.public import PublicKey
from nacl.signing import VerifyKey

# Re-exported, not redefined: these are protocol-wide limits and they live in
# src.protocol.limits because more than one module needs them. They stay
# importable from here because this is where callers have always found them.
from .limits import MAX_FRAME_BYTES, MAX_VARINT_SHIFT
from .padding import DEFAULT_PROFILE, PaddingProfile

#: Wire format version. Independent of PROTOCOL_VERSION: the session protocol
#: version governs key derivation, this one governs byte layout. A session
#: survives a wire format change.
#:
#: Version 3 added post-quantum support. A prekey bundle carries a one-time
#: ML-KEM-768 encapsulation key and its id; an initial handshake carries the
#: matching 1088-byte ciphertext and its id, both inside the initiator's signed
#: transcript. Version 1 and 2 bundles are refused with :class:`UnsupportedVersion`
#: rather than decoded, which matters more than usual here: a v1 plaintext has
#: no length prefix, so unpadding one would read its first byte as a length and
#: could return a plausible-looking fragment instead of failing. Inside the AEAD
#: that prefix cannot be forged by an attacker, so the only way to reach that
#: state is a version mismatch -- which is exactly what the version byte is for.
WIRE_VERSION = 3

#: Post-quantum field widths, from src.crypto.pq. Duplicated as local names
#: rather than imported because this module's rule is that a decoder pins its own
#: expected sizes: an imported constant that changed would change what the
#: decoder accepts, and the point of the check is to notice that the wire moved.
PQ_PUBLIC_KEY_LEN = 1184
PQ_CIPHERTEXT_LEN = 1088

KEY_LEN = 32
SIG_LEN = 64
_MAC_LEN = 32
_UINT32_MAX = 2 ** 32
_UINT64_MAX = (1 << 64) - 1

#: AEAD framing inside a ratchet message. A ciphertext shorter than nonce plus
#: tag cannot be a message, and the ratchet would reject it; the wire refuses it
#: first so the error names the real problem.
_NONCE_LEN = 24
_TAG_LEN = 16
_MIN_CIPHERTEXT = _NONCE_LEN + _TAG_LEN


class WireError(Exception):
    """Base class for every wire decoding failure."""


class TruncatedFrame(WireError):
    """Input ended before a required field was complete."""


class TrailingData(WireError):
    """Input contained bytes after a complete frame."""


class InvalidVarint(WireError):
    """A length or counter used a non-canonical or oversized varint."""


class FrameTooLarge(WireError):
    """The declared payload length exceeds the limit."""


class UnsupportedVersion(WireError):
    """The frame declares a wire version this build does not speak."""


class UnknownMessageKind(WireError):
    """The frame declares a message type this build does not know."""


class InvalidField(WireError):
    """A field decoded correctly but holds an unusable value."""


# ---------------------------------------------------------------------------
# varint
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """
    Encode an unsigned integer as a canonical LEB128 varint.

    Raises:
        ValueError: negative or larger than 2**64-1.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError('varint value must be an int')
    if value < 0 or value > _UINT64_MAX:
        raise ValueError(f'varint out of range: {value}')
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    """
    Decode a canonical LEB128 varint from ``buf`` at ``offset``.

    Returns:
        ``(value, next_offset)``

    Raises:
        TruncatedFrame: the varint continues past the end of the input.
        InvalidVarint: the encoding is not minimal, or exceeds 64 bits.
    """
    result = 0
    shift = 0
    start = offset
    last_byte = 0
    while True:
        if offset >= len(buf):
            raise TruncatedFrame('varint runs past end of input')
        last_byte = buf[offset]
        offset += 1
        result |= (last_byte & 0x7F) << shift
        if not last_byte & 0x80:
            break
        shift += 7
        if shift > MAX_VARINT_SHIFT:
            raise InvalidVarint('varint longer than 64 bits')
    if offset - start > 1 and last_byte == 0x00:
        # e.g. 0x80 0x00 encodes 0 in two bytes; the canonical form is 0x00.
        raise InvalidVarint('non-canonical varint: redundant trailing byte')
    return result, offset


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def _read_exact(buf: bytes, offset: int, count: int, what: str) -> Tuple[bytes, int]:
    end = offset + count
    if end > len(buf):
        raise TruncatedFrame(
            f'need {count} bytes for {what}, have {len(buf) - offset}'
        )
    return buf[offset:end], end


def _encode_frame(kind: int, payload: bytes) -> bytes:
    if len(payload) > MAX_FRAME_BYTES:
        raise FrameTooLarge(
            f'payload of {len(payload)} bytes exceeds {MAX_FRAME_BYTES}'
        )
    return bytes((WIRE_VERSION, kind)) + encode_varint(len(payload)) + payload


def _decode_header(buf: bytes, expected_kinds: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Validate and strip the frame header.

    Returns:
        ``(kind, payload_offset)``

    Raises:
        TruncatedFrame, UnsupportedVersion, UnknownMessageKind, FrameTooLarge
    """
    if len(buf) < 3:
        raise TruncatedFrame('frame shorter than its header')
    version = buf[0]
    if version != WIRE_VERSION:
        raise UnsupportedVersion(
            f'wire version {version} != {WIRE_VERSION}'
        )
    kind = buf[1]
    if kind not in expected_kinds:
        raise UnknownMessageKind(f'unknown message kind 0x{kind:02x}')
    length, offset = decode_varint(buf, 2)
    if length > MAX_FRAME_BYTES:
        raise FrameTooLarge(
            f'declared payload of {length} bytes exceeds {MAX_FRAME_BYTES}'
        )
    return kind, offset


def _extract_payload(buf: bytes, offset: int) -> bytes:
    length, offset = decode_varint(buf, 2)
    payload, end = _read_exact(buf, offset, length, 'payload')
    if end != len(buf):
        raise TrailingData(f'{len(buf) - end} unexpected trailing bytes')
    return payload


# ---------------------------------------------------------------------------
# message kinds
# ---------------------------------------------------------------------------

KIND_PREKEY_BUNDLE = 0x01
KIND_HANDSHAKE_INIT = 0x02
KIND_RATCHET_MESSAGE = 0x03
KIND_KEY_CONFIRMATION = 0x04


# ---------------------------------------------------------------------------
# PreKeyBundle
# ---------------------------------------------------------------------------

def encode_prekey_bundle(bundle: PreKeyBundle) -> bytes:
    """
    Encode a prekey bundle for transmission.

    The caller is responsible for having authenticated the bundle; encoding
    proves nothing.

    The post-quantum encapsulation key is 1184 bytes, so a bundle with one is
    about 1.4 KiB. That is the price of the property, on every handshake, and it
    is worth stating as a size rather than leaving it to be discovered in a
    profile.
    """
    payload = bytearray()
    payload += bytes(bundle.identity_key)
    payload += bytes(bundle.signed_prekey)
    payload += bundle.signed_prekey_signature
    if bundle.one_time_prekey is None:
        payload.append(0x00)
    else:
        payload.append(0x01)
        payload += bytes(bundle.one_time_prekey)
        payload += encode_varint(bundle.one_time_prekey_id)
    payload += encode_varint(bundle.signed_prekey_id)
    if bundle.pq_prekey is None:
        payload.append(0x00)
    else:
        payload.append(0x01)
        payload += bundle.pq_prekey.public_bytes_raw()
        payload += encode_varint(bundle.pq_prekey_id)
    return _encode_frame(KIND_PREKEY_BUNDLE, bytes(payload))


def decode_prekey_bundle(data: bytes) -> PreKeyBundle:
    """
    Decode a prekey bundle.

    Raises:
        WireError: on any malformation. The returned bundle is not
            authenticated; call ``X3DH.verify_bundle`` on it.
        InvalidField: the post-quantum key is present but not a usable
            ML-KEM-768 encapsulation key.
    """
    _decode_header(data, (KIND_PREKEY_BUNDLE,))
    payload = _extract_payload(data, 0)

    expected_min = KEY_LEN + KEY_LEN + SIG_LEN + 1 + 1 + 1
    if len(payload) < expected_min:
        raise TruncatedFrame(
            f'bundle payload of {len(payload)} bytes, minimum {expected_min}'
        )

    offset = 0
    identity_key, offset = _read_exact(payload, offset, KEY_LEN, 'identity_key')
    signed_prekey, offset = _read_exact(payload, offset, KEY_LEN, 'signed_prekey')
    signature, offset = _read_exact(payload, offset, SIG_LEN, 'signature')

    has_one_time = payload[offset]
    offset += 1
    if has_one_time not in (0x00, 0x01):
        raise InvalidField(f'one_time_prekey flag must be 0 or 1, got {has_one_time}')

    one_time_prekey = None
    one_time_prekey_id = None
    if has_one_time:
        raw, offset = _read_exact(payload, offset, KEY_LEN, 'one_time_prekey')
        one_time_prekey = PublicKey(raw)
        one_time_prekey_id, offset = decode_varint(payload, offset)
        if one_time_prekey_id > _UINT64_MAX:
            raise InvalidVarint('one_time_prekey_id out of range')

    signed_prekey_id, offset = decode_varint(payload, offset)

    has_pq = payload[offset]
    offset += 1
    if has_pq not in (0x00, 0x01):
        raise InvalidField(f'pq_prekey flag must be 0 or 1, got {has_pq}')

    pq_prekey = None
    pq_prekey_id = None
    if has_pq:
        raw, offset = _read_exact(
            payload, offset, PQ_PUBLIC_KEY_LEN, 'pq_prekey'
        )
        # Loaded through the pq module, which checks the length and the type.
        # A DER SPKI would be the wrong choice here: load_der_public_key accepts
        # an Ed25519 or X25519 SPKI and returns that type instead, so a bundle
        # carrying a valid key of the wrong algorithm would decode into an
        # object that looks like a key and is not one.
        try:
            pq_prekey = load_pq_public_key(raw)
        except PQError as exc:
            raise InvalidField(f'pq_prekey is not usable: {exc}') from exc
        pq_prekey_id, offset = decode_varint(payload, offset)
        if pq_prekey_id > _UINT64_MAX:
            raise InvalidVarint('pq_prekey_id out of range')

    if offset != len(payload):
        raise TrailingData(f'{len(payload) - offset} unexpected trailing bytes')

    try:
        return PreKeyBundle(
            identity_key=VerifyKey(identity_key),
            signed_prekey=PublicKey(signed_prekey),
            signed_prekey_signature=signature,
            one_time_prekey=one_time_prekey,
            one_time_prekey_id=one_time_prekey_id,
            signed_prekey_id=signed_prekey_id,
            pq_prekey=pq_prekey,
            pq_prekey_id=pq_prekey_id,
        )
    except BadSignatureError as exc:  # pragma: no cover - guarded above
        raise InvalidField('malformed key material') from exc


# ---------------------------------------------------------------------------
# HandshakeInit
# ---------------------------------------------------------------------------

def encode_handshake_init(init: HandshakeInit) -> bytes:
    """
    Encode an initiator handshake.

    ``identity_key_x25519`` is deliberately not transmitted: the receiver
    derives it from ``identity_key`` and the initiator's signature covers the
    derived value, so putting it on the wire would only create a second copy
    that can disagree with the first.

    The post-quantum ciphertext is transmitted, because the responder has no
    other way to learn it. It is inside the initiator's signed transcript, so it
    cannot be swapped for one the attacker generated itself.
    """
    payload = bytearray()
    payload += bytes(init.identity_key)
    payload += bytes(init.ephemeral_key)
    payload += init.signature
    if init.one_time_prekey_id is None:
        payload.append(0x00)
    else:
        payload.append(0x01)
        payload += encode_varint(init.one_time_prekey_id)
    if init.pq_ciphertext is None:
        payload.append(0x00)
    else:
        payload.append(0x01)
        payload += init.pq_ciphertext
        payload += encode_varint(init.pq_prekey_id)
    return _encode_frame(KIND_HANDSHAKE_INIT, bytes(payload))


def decode_handshake_init(data: bytes) -> HandshakeInit:
    """
    Decode an initiator handshake, re-deriving the X25519 identity key.

    Raises:
        WireError: on any malformation.
        InvalidField: the post-quantum ciphertext is present but the wrong
            length for ML-KEM-768.
    """
    _decode_header(data, (KIND_HANDSHAKE_INIT,))
    payload = _extract_payload(data, 0)

    expected_min = KEY_LEN + KEY_LEN + SIG_LEN + 1 + 1
    if len(payload) < expected_min:
        raise TruncatedFrame(
            f'handshake payload of {len(payload)} bytes, minimum {expected_min}'
        )

    offset = 0
    identity_key, offset = _read_exact(payload, offset, KEY_LEN, 'identity_key')
    ephemeral_key, offset = _read_exact(payload, offset, KEY_LEN, 'ephemeral_key')
    signature, offset = _read_exact(payload, offset, SIG_LEN, 'signature')

    has_opk = payload[offset]
    offset += 1
    if has_opk not in (0x00, 0x01):
        raise InvalidField(f'one_time_prekey flag must be 0 or 1, got {has_opk}')

    one_time_prekey_id = None
    if has_opk:
        one_time_prekey_id, offset = decode_varint(payload, offset)

    has_pq = payload[offset]
    offset += 1
    if has_pq not in (0x00, 0x01):
        raise InvalidField(f'pq_ciphertext flag must be 0 or 1, got {has_pq}')

    pq_ciphertext = None
    pq_prekey_id = None
    if has_pq:
        raw, offset = _read_exact(
            payload, offset, PQ_CIPHERTEXT_LEN, 'pq_ciphertext'
        )
        pq_ciphertext = raw
        pq_prekey_id, offset = decode_varint(payload, offset)
        if pq_prekey_id > _UINT64_MAX:
            raise InvalidVarint('pq_prekey_id out of range')

    if offset != len(payload):
        raise TrailingData(f'{len(payload) - offset} unexpected trailing bytes')

    verify_key = VerifyKey(identity_key)
    try:
        derived = identity_public_x25519(verify_key)
    except Exception as exc:
        # A tampered or non-canonical identity key is not a curve point, and
        # libsodium signals that with its own exception types. Untrusted input
        # must not get to choose which exception escapes, so it all becomes
        # InvalidField.
        raise InvalidField('identity_key is not a usable Ed25519 point') from exc

    try:
        return HandshakeInit(
            identity_key=verify_key,
            # Derived, never transmitted: a tampered copy cannot be substituted.
            identity_key_x25519=derived,
            ephemeral_key=PublicKey(ephemeral_key),
            signature=signature,
            one_time_prekey_id=one_time_prekey_id,
            pq_ciphertext=pq_ciphertext,
            pq_prekey_id=pq_prekey_id,
        )
    except PQError as exc:
        # Length disagreement between the encoder's constant and the dataclass's
        # check, or a hostile id. Either way the frame is unusable.
        raise InvalidField(f'malformed handshake: {exc}') from exc


# ---------------------------------------------------------------------------
# ratchet message
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireMessage:
    """
    One ratchet message as it appears on the wire.

    ``dh``, ``pn`` and ``n`` are the authenticated header. They travel in the
    clear because the receiver needs them to derive the message key, and they
    are bound to the ciphertext as AEAD associated data by
    :class:`~src.crypto.double_ratchet.DoubleRatchet`, so they cannot be
    modified without the message failing to decrypt.

    ``ciphertext`` is the authenticated part only. Whatever padding follows it
    in the frame is framing, not content: see :func:`encode_ratchet_message`.
    """

    dh: bytes
    pn: int
    n: int
    ciphertext: bytes

    def to_header(self) -> dict:
        """Return the header dict the ratchet expects."""
        return {'dh': self.dh.hex(), 'pn': self.pn, 'n': self.n}


def _quantised_payload_length(target: int, minimum: int) -> int:
    """
    The payload length that makes a frame exactly ``target`` bytes long.

    A frame is ``version | kind | varint(payload_len) | payload``, and the
    varint's own width depends on the payload length it announces, so the sum
    has to be solved rather than computed. Enumerating the possible varint
    widths is exact and cheap: there are at most nine of them.

    Not used by the encoder, which pads the payload to a ladder instead and lets
    the frame length follow from it. It stays because it is the only way to ask
    the question "what payload length makes this frame exactly N bytes", and the
    answer is not the same as ``N - 2``.

    Raises:
        FrameTooLarge: no payload length reaches the target, which happens only
            if the target is smaller than a frame's fixed parts.
    """
    for width in range(1, 10):
        payload_length = target - 2 - width
        if payload_length < 0:
            break
        if len(encode_varint(payload_length)) == width:
            if payload_length < minimum:
                raise FrameTooLarge(
                    f'frame target of {target} bytes leaves {payload_length} for '
                    f'the payload, below the {minimum} this message needs'
                )
            return payload_length
    raise FrameTooLarge(f'no frame of exactly {target} bytes can be built')


def encode_ratchet_message(ciphertext: bytes, header: dict,
                           profile: Optional[PaddingProfile] = None) -> bytes:
    """
    Encode ``nonce || ciphertext`` together with its header.

    The payload is::

        dh | varint(pn) | varint(n) | varint(ciphertext_len) | ciphertext | fill

    The ciphertext length prefix is what lets the receiver find the end of the
    ciphertext while the frame is quantised. The trailing fill is what makes the
    *frame* land on a bucket, so that two messages of different sizes are the
    same size on the path.

    The fill is not authenticated, and deliberately so. It carries no
    information the receiver needs, and putting it under the AEAD would mean
    re-encrypting it on every counter change for no gain. Changing it is
    harmless: the receiver checks the declared length against the frame, takes
    exactly that many bytes, and the AEAD then fails on the tag because a
    modified length means the tag is no longer the last 16 bytes of what it
    decrypts. That is a detectable failure, not a substitution.

    Args:
        profile: the size ladder. ``None`` uses
            :data:`~src.protocol.padding.DEFAULT_PROFILE`; a profile with a
            floor of zero produces unpadded frames, which is only useful for
            measuring what padding costs.

    Raises:
        WireError: the header is malformed, so encoding it would produce a
            frame the peer must reject.
        PaddingError: the message does not fit in a frame once padded.
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    dh_hex = header.get('dh')
    if not isinstance(dh_hex, str) or len(dh_hex) != KEY_LEN * 2:
        raise InvalidField('header dh must be 32-byte hex')
    try:
        dh = bytes.fromhex(dh_hex)
    except ValueError as exc:
        raise InvalidField('header dh is not valid hex') from exc

    for name in ('pn', 'n'):
        value = header.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidField(f'header {name} must be an int')
        if not 0 <= value < _UINT32_MAX:
            raise InvalidField(f'header {name} out of range')

    if not isinstance(ciphertext, (bytes, bytearray, memoryview)):
        raise InvalidField('ciphertext must be bytes')
    ciphertext = bytes(ciphertext)

    prefix = (
        dh
        + encode_varint(header['pn'])
        + encode_varint(header['n'])
        + encode_varint(len(ciphertext))
    )
    content = len(prefix) + len(ciphertext)
    if not profile.enabled:
        return _encode_frame(KIND_RATCHET_MESSAGE, prefix + ciphertext)

    # The payload is padded up to the profile's frame ladder, which sits
    # FRAME_OVERHEAD_MAX above the plaintext ladder so that a short message
    # costs one bucket rather than two.
    target_payload = profile.frame_payload_length(content)
    if target_payload < content:  # pragma: no cover - the ladder guarantees it
        raise FrameTooLarge(
            f'padded payload of {target_payload} is smaller than the {content} '
            f'bytes of message'
        )
    fill = os.urandom(target_payload - content)
    return _encode_frame(KIND_RATCHET_MESSAGE, prefix + ciphertext + fill)


def decode_ratchet_message(data: bytes) -> WireMessage:
    """
    Decode a ratchet message.

    Trailing bytes after the declared ciphertext are framing padding and are
    discarded. They are not required to be a whole number of buckets: a peer
    with a different padding profile is not an error, and enforcing one here
    would make a padding change into a hard protocol break.

    Raises:
        WireError: on any malformation.
    """
    _decode_header(data, (KIND_RATCHET_MESSAGE,))
    payload = _extract_payload(data, 0)

    # A message is dh, three varints and a ciphertext that at least holds a
    # nonce and a tag. The ciphertext length itself is checked below, against
    # the value the frame declares.
    minimum = KEY_LEN + 3 + _MIN_CIPHERTEXT
    if len(payload) < minimum:
        raise TruncatedFrame(
            f'ratchet payload of {len(payload)} bytes, minimum {minimum}'
        )

    offset = 0
    dh, offset = _read_exact(payload, offset, KEY_LEN, 'header dh')
    pn, offset = decode_varint(payload, offset)
    n, offset = decode_varint(payload, offset)
    ciphertext_length, offset = decode_varint(payload, offset)
    if pn > _UINT32_MAX or n > _UINT32_MAX:
        raise InvalidField('header counter exceeds 32 bits')
    if ciphertext_length > _UINT32_MAX:
        raise InvalidField('ciphertext length exceeds 32 bits')
    if ciphertext_length < _MIN_CIPHERTEXT:
        # Checked against the declared ciphertext, not the payload. A padded
        # frame is large whatever it carries, so a length check on the frame
        # would pass a ciphertext too short to hold a nonce and a tag, and the
        # failure would surface later as a confusing AEAD error instead of
        # here.
        raise InvalidField(
            f'ciphertext of {ciphertext_length} bytes cannot hold a '
            f'{_NONCE_LEN}-byte nonce and a {_TAG_LEN}-byte tag'
        )

    ciphertext, end = _read_exact(
        payload, offset, ciphertext_length, 'ciphertext'
    )
    # Anything left is the length padding the sender added to quantise the
    # frame. Not an error if present, and not required to be.
    if end > len(payload):  # pragma: no cover - _read_exact already checks
        raise TruncatedFrame('ciphertext runs past the end of the frame')
    return WireMessage(
        dh=dh, pn=pn, n=n, ciphertext=bytes(ciphertext)
    )


def protocol_version() -> int:
    """Session protocol version this build speaks, for the handshake transcript."""
    return PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# key confirmation
# ---------------------------------------------------------------------------

def encode_key_confirmation(confirmation: KeyConfirmation) -> bytes:
    """
    Encode a responder's key confirmation.

    Carrying the MAC in the clear is intended: it is a MAC, not a secret, and
    it is useless without the root key that keys it.
    """
    payload = bytes(confirmation.ephemeral_key) + confirmation.mac
    return _encode_frame(KIND_KEY_CONFIRMATION, payload)


def decode_key_confirmation(data: bytes) -> KeyConfirmation:
    """
    Decode a key confirmation.

    Raises:
        WireError: on any malformation. Whether the MAC is *correct* is decided
            by :meth:`~src.crypto.x3dh.X3DHSession.verify_key_confirmation`,
            not here.
    """
    _decode_header(data, (KIND_KEY_CONFIRMATION,))
    payload = _extract_payload(data, 0)

    expected = KEY_LEN + _MAC_LEN
    if len(payload) != expected:
        raise TruncatedFrame(
            f'confirmation payload of {len(payload)} bytes, expected {expected}'
        )

    try:
        return KeyConfirmation(
            ephemeral_key=PublicKey(payload[:KEY_LEN]),
            mac=payload[KEY_LEN:],
        )
    except KeyConfirmationFailed as exc:
        # Structural validation only; KeyConfirmation.__post_init__ re-checks
        # the lengths, which the slice above already guarantees.
        raise InvalidField('malformed key confirmation') from exc
