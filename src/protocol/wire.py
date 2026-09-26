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
from typing import Tuple

from ..crypto.double_ratchet import PROTOCOL_VERSION
from ..crypto.x3dh import HandshakeInit, PreKeyBundle
from ..crypto.key_management import identity_public_x25519
from nacl.exceptions import BadSignatureError
from nacl.public import PublicKey
from nacl.signing import VerifyKey

#: Wire format version. Independent of PROTOCOL_VERSION: the session protocol
#: version governs key derivation, this one governs byte layout. A session
#: survives a wire format change.
WIRE_VERSION = 1

#: Refuse frames larger than this before allocating. Ratchet messages are small;
#: anything near this limit is an attack or a bug.
MAX_FRAME_BYTES = 1 << 20  # 1 MiB

#: Guard against absurd varints even inside a legal frame.
MAX_VARINT_SHIFT = 63

KEY_LEN = 32
SIG_LEN = 64
_UINT32_MAX = 2 ** 32
_UINT64_MAX = (1 << 64) - 1


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


# ---------------------------------------------------------------------------
# PreKeyBundle
# ---------------------------------------------------------------------------

def encode_prekey_bundle(bundle: PreKeyBundle) -> bytes:
    """
    Encode a prekey bundle for transmission.

    The caller is responsible for having authenticated the bundle; encoding
    proves nothing.
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
    return _encode_frame(KIND_PREKEY_BUNDLE, bytes(payload))


def decode_prekey_bundle(data: bytes) -> PreKeyBundle:
    """
    Decode a prekey bundle.

    Raises:
        WireError: on any malformation. The returned bundle is not
            authenticated; call ``X3DH.verify_bundle`` on it.
    """
    _decode_header(data, (KIND_PREKEY_BUNDLE,))
    payload = _extract_payload(data, 0)

    expected_min = KEY_LEN + KEY_LEN + SIG_LEN + 1
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

    if offset != len(payload):
        raise TrailingData(f'{len(payload) - offset} unexpected trailing bytes')

    try:
        return PreKeyBundle(
            identity_key=VerifyKey(identity_key),
            signed_prekey=PublicKey(signed_prekey),
            signed_prekey_signature=signature,
            one_time_prekey=one_time_prekey,
            one_time_prekey_id=one_time_prekey_id,
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
    return _encode_frame(KIND_HANDSHAKE_INIT, bytes(payload))


def decode_handshake_init(data: bytes) -> HandshakeInit:
    """
    Decode an initiator handshake, re-deriving the X25519 identity key.

    Raises:
        WireError: on any malformation.
    """
    _decode_header(data, (KIND_HANDSHAKE_INIT,))
    payload = _extract_payload(data, 0)

    expected_min = KEY_LEN + KEY_LEN + SIG_LEN + 1
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

    return HandshakeInit(
        identity_key=verify_key,
        # Derived, never transmitted: a tampered copy cannot be substituted.
        identity_key_x25519=derived,
        ephemeral_key=PublicKey(ephemeral_key),
        signature=signature,
        one_time_prekey_id=one_time_prekey_id,
    )


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
    """

    dh: bytes
    pn: int
    n: int
    ciphertext: bytes

    def to_header(self) -> dict:
        """Return the header dict the ratchet expects."""
        return {'dh': self.dh.hex(), 'pn': self.pn, 'n': self.n}


def encode_ratchet_message(ciphertext: bytes, header: dict) -> bytes:
    """
    Encode ``nonce || ciphertext`` together with its header.

    Raises:
        WireError: the header is malformed, so encoding it would produce a
            frame the peer must reject.
    """
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

    payload = bytes((
        dh
        + encode_varint(header['pn'])
        + encode_varint(header['n'])
        + ciphertext
    ))
    return _encode_frame(KIND_RATCHET_MESSAGE, payload)


def decode_ratchet_message(data: bytes) -> WireMessage:
    """
    Decode a ratchet message.

    Raises:
        WireError: on any malformation.
    """
    _decode_header(data, (KIND_RATCHET_MESSAGE,))
    payload = _extract_payload(data, 0)

    # A message is dh + at least two one-byte varints + nonce + tag.
    minimum = KEY_LEN + 2 + 24 + 16
    if len(payload) < minimum:
        raise TruncatedFrame(
            f'ratchet payload of {len(payload)} bytes, minimum {minimum}'
        )

    offset = 0
    dh, offset = _read_exact(payload, offset, KEY_LEN, 'header dh')
    pn, offset = decode_varint(payload, offset)
    n, offset = decode_varint(payload, offset)
    ciphertext = payload[offset:]

    if pn > _UINT32_MAX or n > _UINT32_MAX:
        raise InvalidField('header counter exceeds 32 bits')

    return WireMessage(dh=dh, pn=pn, n=n, ciphertext=bytes(ciphertext))


def protocol_version() -> int:
    """Session protocol version this build speaks, for the handshake transcript."""
    return PROTOCOL_VERSION
