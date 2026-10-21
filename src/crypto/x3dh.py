"""
X3DH (Extended Triple Diffie-Hellman) implementation.

Specification: https://signal.org/docs/specifications/x3dh/

Security notes
--------------
The property that makes X3DH worth using is *authentication of the prekey
bundle*: the initiator must verify that the signed prekey really was signed by
the identity key it is advertised under, and the responder must verify that the
initiator owns the identity key it presents. Without both checks X3DH degrades
to unauthenticated Diffie-Hellman, and an active attacker substitutes keys and
reads the session.

Both checks are implemented here and cannot be skipped:

* :meth:`X3DH.initiate_handshake` verifies ``signed_prekey_signature`` against
  ``identity_key`` before any key from the bundle is used.
* :meth:`X3DH.receive_handshake` accepts a :class:`HandshakeInit` whose
  signature covers a length-prefixed transcript, and verifies it against the
  identity key carried *inside* that structure. The caller's word about who the
  initiator is no longer trusted.

One-time prekeys are single use. :class:`X3DHResponder` enforces that through an
atomic ``consume`` on a :class:`~src.crypto.prekey_store.PreKeyStore`.

Deliberate deviations from the specification are documented at the points where
they occur.
"""

import hmac
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import nacl.bindings
from nacl.exceptions import BadSignatureError, CryptoError
from nacl.public import PrivateKey, PublicKey
from nacl.signing import SigningKey, VerifyKey

from .kdf import extract as kdf_extract
from .kdf import expand as kdf_expand
from .key_management import identity_private_x25519, identity_public_x25519
from .pq import (
    CIPHERTEXT_LEN as PQ_CIPHERTEXT_LEN,
    NoSuchPQPreKey,
    PQError,
    PQPreKeyExhausted,
    PQPreKeyStore,
    decapsulate as pq_decapsulate,
    encapsulate as pq_encapsulate,
    generate_prekey as generate_pq_prekey,
    load_public_key as pq_load_public_key,
)
from .prekey_store import OneTimePreKey, PreKeyStore

# Domain separation for the initiator signature transcript.
INIT_TRANSCRIPT_LABEL = b"X3DH-init-v1"

# Default context string mixed into the HKDF ``info`` field; see
# :meth:`X3DH.derive_master_key`.
DEFAULT_CONTEXT = b"DecentralizedMessenger-X3DH-v1"

# X3DH section 4: for 256-bit curves, F is 0xFF repeated 32 times.
X25519_F = b'\xff' * 32

# Key confirmation, X3DH section 4.3. The MAC is keyed by a value derived from
# the shared root key, so only a peer that derived the same root key can produce
# or verify it.
KEY_CONFIRMATION_INFO = b"WhisperKeyConfirmation"

_DH_LEN = 32
_SIG_LEN = 64
_MAC_LEN = 32


class X3DHError(Exception):
    """Base class for X3DH failures."""


class InvalidHandshakeError(X3DHError):
    """Raised when a handshake structure is malformed or inconsistent."""


class InvalidSignatureError(InvalidHandshakeError):
    """
    Raised when a prekey bundle or handshake signature does not verify.

    A subset of :class:`InvalidHandshakeError`, so a caller that only wants to
    reject bad handshakes can catch the parent class.
    """


class InvalidKeyError(X3DHError):
    """Raised for degenerate keys, for example a low-order point."""


class PrekeyPoolExhausted(X3DHError):
    """
    Raised when a bundle is requested but the one-time prekey pool is empty.

    A responder with an empty pool can still run the 3-DH variant, which works
    but has no forward secrecy. Returning such a bundle by default would make
    the downgrade silent, and the peer would have no way to notice its session
    is weaker than intended. Callers must opt in explicitly.
    """


class RetiredSignedPrekey(X3DHError):
    """
    Raised when a bundle's signed prekey predates the caller's minimum.

    Peers that cached a bundle keep a valid signature over it after a rotation,
    so the signature alone cannot distinguish a current prekey from a retired
    one. The monotonic ``signed_prekey_id`` lets a client that has learned of a
    rotation refuse the old one.
    """


class KeyConfirmationRequired(X3DHError):
    """
    Raised when the root key is requested before key confirmation.

    Without this, an initiator cannot tell a real peer from a responder that
    derived a different secret and simply said nothing: the handshake would
    look successful and every message would fail later, or worse, succeed under
    a key the peer does not hold.
    """


class KeyConfirmationFailed(X3DHError):
    """Raised when the peer's key confirmation MAC does not verify."""


def _u32(value: int) -> bytes:
    """Big-endian 32-bit length prefix, for transcript canonicalisation."""
    return value.to_bytes(4, 'big')


def length_prefixed(*chunks: Optional[bytes]) -> bytes:
    """
    Concatenate chunks with explicit length prefixes.

    Length prefixing keeps the encoding unambiguous, so no two different field
    sets can produce the same byte string. Without it a signature could be
    replayed against a differently split set of fields.

    Public, because :mod:`src.protocol.session` builds its AEAD associated data
    out of the same kind of object: an ordered set of fields that has to be
    canonically encoded before it can be hashed. Two implementations of a
    canonicalisation primitive on either side of a MAC are precisely the
    failure this repository already paid for once, when the X3DH and Double
    Ratchet call sites carried separate copies of HKDF and one of them was
    wrong in a way that no round-trip test noticed (see :mod:`src.crypto.kdf`).
    """
    out = bytearray()
    for chunk in chunks:
        if chunk is None:
            out += _u32(0)
        else:
            out += _u32(len(chunk))
            out += chunk
    return bytes(out)


def _init_transcript(
    identity_key: VerifyKey,
    identity_key_x25519: PublicKey,
    ephemeral_key: PublicKey,
    one_time_prekey_id: Optional[int],
    pq_ciphertext: Optional[bytes],
    pq_prekey_id: Optional[int],
    associated_data: bytes,
) -> bytes:
    """
    Canonical transcript covered by the initiator's handshake signature.

    Binds the identity key, both of its curve representations, the ephemeral
    key, the referenced prekey id, the post-quantum ciphertext and the
    post-quantum prekey id, and any external associated data.

    The post-quantum ciphertext is in here and that is the whole reason the
    handshake carries it in the clear. A ciphertext that is not covered by the
    signature can be replaced by an active attacker with one the attacker
    generated itself, and then the attacker knows the ML-KEM secret while the two
    honest parties do not share it. Covering it costs 1088 bytes of hashing per
    handshake and removes the cheapest possible post-quantum downgrade.
    """
    opk = (None if one_time_prekey_id is None
           else one_time_prekey_id.to_bytes(8, 'big'))
    pq_id = (None if pq_prekey_id is None
             else pq_prekey_id.to_bytes(8, 'big'))
    return length_prefixed(
        INIT_TRANSCRIPT_LABEL,
        bytes(identity_key),
        bytes(identity_key_x25519),
        bytes(ephemeral_key),
        opk,
        pq_ciphertext,
        pq_id,
        associated_data,
    )


@dataclass(frozen=True)
class PreKeyBundle:
    """
    The prekey bundle published by a responder.

    ``signed_prekey_signature`` must be an Ed25519 signature over the raw
    32-byte ``signed_prekey``, as produced by
    :meth:`X3DH.generate_signed_prekey`.

    ``pq_prekey`` is a one-time ML-KEM-768 encapsulation key. It is *not*
    optional in practice: a bundle without one produces a session whose forward
    secrecy does not survive a future break of X25519, and
    :meth:`X3DHResponder.publish_bundle` refuses to hand one out unless the
    caller says explicitly that a non-post-quantum session is acceptable.

    The post-quantum key is 1184 bytes and the X25519 material is 128, so a
    bundle is roughly ten times larger than it was. That is the cost of the
    property, paid in bytes on every handshake.
    """

    identity_key: VerifyKey
    signed_prekey: PublicKey
    signed_prekey_signature: bytes
    one_time_prekey: Optional[PublicKey] = None
    one_time_prekey_id: Optional[int] = None
    #: Monotonic counter identifying the current signed prekey. Zero means
    #: "not tracked", which leaves the signature as the only check.
    signed_prekey_id: int = 0
    #: One-time ML-KEM-768 encapsulation key, and the id it is published under.
    pq_prekey: Optional[object] = None
    pq_prekey_id: Optional[int] = None

    def __post_init__(self) -> None:
        if len(bytes(self.identity_key)) != _DH_LEN:
            raise InvalidHandshakeError('identity key must be 32 bytes')
        if len(bytes(self.signed_prekey)) != _DH_LEN:
            raise InvalidHandshakeError('signed prekey must be 32 bytes')
        if len(self.signed_prekey_signature) != _SIG_LEN:
            raise InvalidSignatureError(
                f'signed prekey signature must be {_SIG_LEN} bytes'
            )
        if (self.one_time_prekey is None) != (self.one_time_prekey_id is None):
            raise InvalidHandshakeError(
                'one_time_prekey and one_time_prekey_id must both be set or unset'
            )
        if self.one_time_prekey is not None and len(bytes(self.one_time_prekey)) != _DH_LEN:
            raise InvalidHandshakeError('one-time prekey must be 32 bytes')
        if isinstance(self.signed_prekey_id, bool) or \
                not isinstance(self.signed_prekey_id, int) or \
                self.signed_prekey_id < 0:
            raise InvalidHandshakeError(
                'signed_prekey_id must be a non-negative int'
            )
        if (self.pq_prekey is None) != (self.pq_prekey_id is None):
            raise InvalidHandshakeError(
                'pq_prekey and pq_prekey_id must both be set or unset'
            )
        if self.pq_prekey is not None:
            # Checked through the pq module's loader rather than by isinstance
            # here, so the length and the type are both validated in one place.
            pq_load_public_key(self.pq_prekey.public_bytes_raw())


@dataclass(frozen=True)
class HandshakeInit:
    """
    The initiator's handshake message, carrying its own proof of possession.

    ``signature`` is an Ed25519 signature by ``identity_key`` over
    :meth:`transcript`. This is the only structure :meth:`X3DH.receive_handshake`
    accepts, so a handshake cannot be processed without proof that the sender
    owns the identity key it names.

    ``pq_ciphertext`` is the ML-KEM-768 ciphertext the initiator encapsulated
    against the responder's post-quantum prekey, 1088 bytes. It travels here
    because the responder has no other way to learn it, and it is inside the
    signed transcript because otherwise an active attacker could replace it with
    a ciphertext of its own choosing: the initiator would still have produced a
    real ciphertext, the attacker would decapsulate it, and both sides would
    derive a root key the other does not share -- a downgrade the key
    confirmation would then catch, but only after the fact and with no way to
    tell a quantum adversary from a broken peer.
    """

    identity_key: VerifyKey
    identity_key_x25519: PublicKey
    ephemeral_key: PublicKey
    signature: bytes
    one_time_prekey_id: Optional[int] = None
    pq_ciphertext: Optional[bytes] = None
    pq_prekey_id: Optional[int] = None

    def __post_init__(self) -> None:
        for name, value in (
            ('identity_key', self.identity_key),
            ('identity_key_x25519', self.identity_key_x25519),
            ('ephemeral_key', self.ephemeral_key),
        ):
            if len(bytes(value)) != _DH_LEN:
                raise InvalidHandshakeError(f'{name} must be {_DH_LEN} bytes')
        if len(self.signature) != _SIG_LEN:
            raise InvalidSignatureError(
                f'handshake signature must be {_SIG_LEN} bytes'
            )
        if self.one_time_prekey_id is not None and not (
            0 <= self.one_time_prekey_id < 2 ** 64
        ):
            raise InvalidHandshakeError('one_time_prekey_id out of range')
        if (self.pq_ciphertext is None) != (self.pq_prekey_id is None):
            raise InvalidHandshakeError(
                'pq_ciphertext and pq_prekey_id must both be set or unset'
            )
        if self.pq_ciphertext is not None and \
                len(self.pq_ciphertext) != PQ_CIPHERTEXT_LEN:
            raise InvalidHandshakeError(
                f'pq_ciphertext must be {PQ_CIPHERTEXT_LEN} bytes, got '
                f'{len(self.pq_ciphertext)}'
            )
        if self.pq_prekey_id is not None and not (0 <= self.pq_prekey_id < 2 ** 64):
            raise InvalidHandshakeError('pq_prekey_id out of range')

    def transcript(self, associated_data: bytes = b'') -> bytes:
        """Canonical, length-prefixed byte string covered by ``signature``."""
        return _init_transcript(
            self.identity_key,
            self.identity_key_x25519,
            self.ephemeral_key,
            self.one_time_prekey_id,
            self.pq_ciphertext,
            self.pq_prekey_id,
            associated_data,
        )

    def verify(self, associated_data: bytes = b'') -> None:
        """
        Verify the initiator's signature.

        Raises:
            InvalidSignatureError: the signature does not match the transcript.
        """
        if len(self.signature) != _SIG_LEN:
            raise InvalidSignatureError(
                f'handshake signature must be {_SIG_LEN} bytes'
            )
        try:
            self.identity_key.verify(self.transcript(associated_data), self.signature)
        except BadSignatureError as exc:
            raise InvalidSignatureError(
                'initiator handshake signature verification failed'
            ) from exc


@dataclass(frozen=True)
class DhPart:
    """
    One contribution to the shared secret, with the name of what produced it.

    The label is not decoration. Every contribution here is 32 bytes, so a KDF
    handed four of them cannot tell "three X25519 outputs plus a one-time
    prekey" from "three X25519 outputs plus an ML-KEM secret" -- both are four
    32-byte values, and both would produce the same root key from the same
    buffer. That is the ambiguity this type exists to remove: the KDF mixes the
    labels in, so swapping an ML-KEM contribution for an X25519 one changes the
    derived key instead of vanishing into it.
    """

    label: bytes
    value: bytes

    def __post_init__(self) -> None:
        if len(self.value) != _DH_LEN:
            raise InvalidHandshakeError(
                f'DH output must be {_DH_LEN} bytes, got {len(self.value)}'
            )
        if not isinstance(self.label, (bytes, bytearray)):
            raise InvalidHandshakeError('a DH part label must be bytes')


#: Labels for the contributions, in the order they are mixed in. Fixed strings so
#: that the set of combinations a handshake can produce is a closed list rather
#: than whatever a caller passes.
LABEL_IK_SPK = b'DH1'
LABEL_EK_IK = b'DH2'
LABEL_EK_SPK = b'DH3'
LABEL_EK_OPK = b'DH4'
LABEL_KEM = b'PQ'


@dataclass
class X3DHState:
    """
    One side's handshake result.

    The individual contributions are kept as :class:`DhPart` rather than only
    their concatenation, so the KDF step cannot be handed an ambiguous buffer and
    cannot mistake one prekey arrangement for another. Use :meth:`master_key` to
    derive the 32-byte root key for the Double Ratchet.
    """

    dh_parts: Tuple[DhPart, ...]
    ephemeral_key: Optional[PrivateKey] = None
    identity_key: Optional[VerifyKey] = None
    one_time_prekey_id: Optional[int] = None
    pq_prekey_id: Optional[int] = None
    pq_ciphertext: Optional[bytes] = None

    def master_key(self, context: bytes = DEFAULT_CONTEXT) -> bytes:
        return X3DH.derive_master_key(self.dh_parts, context)

    @property
    def is_post_quantum(self) -> bool:
        """Whether the shared secret includes an ML-KEM contribution."""
        return any(part.label == LABEL_KEM for part in self.dh_parts)

    @property
    def shared_secret(self) -> bytes:
        """
        Raw concatenation of the contributions, without F and without the KDF.

        Kept for tests and debugging. Never use it as a message or root key; use
        :meth:`master_key`. The labels are included, so this is not the same
        buffer for two different prekey arrangements.
        """
        return b''.join(
            len(part.label).to_bytes(4, 'big') + part.label + part.value
            for part in self.dh_parts
        )


@dataclass(frozen=True)
class KeyConfirmation:
    """
    The responder's proof that it derived the same root key.

    Carries a fresh ephemeral key, so the confirmation is not replayable into a
    later handshake even if the same root key were ever reused.
    """

    ephemeral_key: PublicKey
    mac: bytes

    def __post_init__(self) -> None:
        if len(bytes(self.ephemeral_key)) != _DH_LEN:
            raise InvalidHandshakeError(
                f'confirmation ephemeral key must be {_DH_LEN} bytes'
            )
        if len(self.mac) != _MAC_LEN:
            raise KeyConfirmationFailed(
                f'confirmation MAC must be {_MAC_LEN} bytes'
            )


def confirmation_mac_key(root_key: bytes) -> bytes:
    """
    Derive the confirmation MAC key from the X3DH root key.

    Binding the MAC to the root key is what makes confirmation meaningful: the
    transcript is public, so without this anyone could produce a valid MAC.
    """
    if len(root_key) != _DH_LEN:
        raise InvalidHandshakeError(f'root key must be {_DH_LEN} bytes')
    return kdf_expand(
        kdf_extract(b'\x00' * _DH_LEN, root_key), KEY_CONFIRMATION_INFO, _MAC_LEN
    )


def confirmation_mac(
    root_key: bytes,
    *,
    associated_data: bytes,
    alice_identity: VerifyKey,
    alice_ephemeral: PublicKey,
    bob_identity: VerifyKey,
    bob_ephemeral: PublicKey,
) -> bytes:
    """
    Compute the key confirmation MAC.

    The transcript binds the external associated data, both identity keys, the
    initiator's ephemeral key and the responder's confirmation ephemeral key, so
    a confirmation cannot be replayed against a different session or a
    substituted identity key.
    """
    transcript = X25519_F + length_prefixed(
        associated_data,
        bytes(alice_identity),
        bytes(alice_ephemeral),
        bytes(bob_identity),
        bytes(identity_public_x25519(bob_identity)),
        bytes(bob_ephemeral),
    )
    return hmac.new(
        confirmation_mac_key(root_key), transcript, hashlib.sha256
    ).digest()


class X3DHSession:
    """
    A handshake whose root key stays locked until the session is confirmed.

    The initiator starts unconfirmed and must verify the responder's
    confirmation MAC before :attr:`root_key` yields anything. The responder
    starts confirmed, because it has nothing to verify: it authenticated the
    initiator and computed the secret itself. It learns that the initiator
    received the confirmation when the first ratchet message decrypts.
    """

    def __init__(
        self,
        state: X3DHState,
        *,
        is_initiator: bool,
        alice_identity: Optional[VerifyKey] = None,
        alice_ephemeral: Optional[PublicKey] = None,
        bob_identity: Optional[VerifyKey] = None,
        associated_data: bytes = b'',
    ) -> None:
        self._state = state
        self.is_initiator = is_initiator
        self._associated_data = associated_data
        self._alice_identity = alice_identity
        self._alice_ephemeral = alice_ephemeral
        self._bob_identity = bob_identity
        self._bob_ephemeral: Optional[PublicKey] = None
        self._is_confirmed = not is_initiator

    @property
    def is_confirmed(self) -> bool:
        return self._is_confirmed

    @property
    def initiator_identity(self) -> VerifyKey:
        """
        Identity key of the side that started the handshake.

        Raises:
            InvalidHandshakeError: the session was built without identities, so
                there is nothing to bind associated data to. Returning None
                would push the failure into the AEAD, where it would be
                indistinguishable from a wrong message key.
        """
        if self._alice_identity is None:
            raise InvalidHandshakeError(
                'this X3DH session carries no initiator identity key'
            )
        return self._alice_identity

    @property
    def responder_identity(self) -> VerifyKey:
        """
        Identity key of the side that answered the handshake.

        Raises:
            InvalidHandshakeError: the session was built without identities.
        """
        if self._bob_identity is None:
            raise InvalidHandshakeError(
                'this X3DH session carries no responder identity key'
            )
        return self._bob_identity

    @property
    def one_time_prekey_id(self) -> Optional[int]:
        return self._state.one_time_prekey_id

    @property
    def pq_prekey_id(self) -> Optional[int]:
        """Which post-quantum prekey this handshake spent, if any."""
        return self._state.pq_prekey_id

    @property
    def is_post_quantum(self) -> bool:
        """
        Whether the root key includes an ML-KEM contribution.

        False means the session's forward secrecy ends the day X25519 is broken,
        however it was configured. Worth asking rather than assuming, and worth
        recording for a session that was accepted on an explicit opt-in.
        """
        return self._state.is_post_quantum

    @property
    def root_key(self) -> bytes:
        """
        The 32-byte root key for the Double Ratchet.

        Raises:
            KeyConfirmationRequired: the initiator has not verified the
                responder's confirmation yet.
        """
        if not self._is_confirmed:
            raise KeyConfirmationRequired(
                'verify the responder key confirmation before using the root key'
            )
        return self._state.master_key()

    def make_key_confirmation(self) -> KeyConfirmation:
        """
        Responder side: produce the confirmation.

        Raises:
            InvalidHandshakeError: called on an initiator session.
        """
        if self.is_initiator:
            raise InvalidHandshakeError(
                'only the responder produces a key confirmation'
            )
        ephemeral = PrivateKey.generate()
        self._bob_ephemeral = ephemeral.public_key
        mac = confirmation_mac(
            self._state.master_key(),
            associated_data=self._associated_data,
            alice_identity=self._alice_identity,
            alice_ephemeral=self._alice_ephemeral,
            bob_identity=self._bob_identity,
            bob_ephemeral=self._bob_ephemeral,
        )
        return KeyConfirmation(ephemeral_key=self._bob_ephemeral, mac=mac)

    def verify_key_confirmation(self, confirmation: KeyConfirmation) -> None:
        """
        Initiator side: check the responder's confirmation and unlock the key.

        Raises:
            InvalidHandshakeError: called on a responder session.
            KeyConfirmationFailed: the MAC does not match.
        """
        if not self.is_initiator:
            raise InvalidHandshakeError(
                'only the initiator verifies a key confirmation'
            )
        if not isinstance(confirmation, KeyConfirmation):
            raise InvalidHandshakeError('confirmation must be a KeyConfirmation')

        expected = confirmation_mac(
            self._state.master_key(),
            associated_data=self._associated_data,
            alice_identity=self._alice_identity,
            alice_ephemeral=self._alice_ephemeral,
            bob_identity=self._bob_identity,
            bob_ephemeral=confirmation.ephemeral_key,
        )
        if not hmac.compare_digest(expected, confirmation.mac):
            raise KeyConfirmationFailed(
                'responder key confirmation MAC does not verify'
            )
        self._bob_ephemeral = confirmation.ephemeral_key
        self._is_confirmed = True


class X3DH:
    """Extended Triple Diffie-Hellman key agreement (Signal specification)."""

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_identity_keys() -> Tuple[SigningKey, VerifyKey]:
        """Generate the long-term Ed25519 identity key pair."""
        signing_key = SigningKey.generate()
        return signing_key, signing_key.verify_key

    @staticmethod
    def generate_signed_prekey(
        identity_key: SigningKey,
    ) -> Tuple[PrivateKey, PublicKey, bytes]:
        """
        Generate a signed prekey.

        Returns:
            ``(spk_private, spk_public, signature)``, where ``signature`` is the
            raw 64-byte Ed25519 signature over ``spk_public.encode()``.
        """
        spk_private = PrivateKey.generate()
        spk_public = spk_private.public_key
        signature = identity_key.sign(spk_public.encode()).signature
        return spk_private, spk_public, signature

    @staticmethod
    def generate_one_time_prekey(key_id: Optional[int] = None) -> OneTimePreKey:
        """Generate a single-use prekey. See :mod:`src.crypto.prekey_store`."""
        private_key = PrivateKey.generate()
        if key_id is None:
            key_id = int.from_bytes(os.urandom(8), 'big')
        return OneTimePreKey(
            key_id=key_id,
            private_key=private_key,
            public_key=private_key.public_key,
        )

    @staticmethod
    def generate_pq_prekey(key_id: Optional[int] = None):
        """
        Generate a single-use ML-KEM-768 prekey.

        See :mod:`src.crypto.pq` for the primitive and, more importantly, for
        why a post-quantum prekey is held to the same single-use discipline as
        the X25519 one.
        """
        return generate_pq_prekey(key_id)

    # ------------------------------------------------------------------
    # Bundle authentication
    # ------------------------------------------------------------------

    @staticmethod
    def verify_bundle(bundle: PreKeyBundle) -> None:
        """
        Verify that ``bundle.signed_prekey`` was signed by ``bundle.identity_key``.

        This is the check whose absence allows an active MITM. It is called by
        :meth:`initiate_handshake` and exposed separately so a caller can
        validate a freshly fetched bundle before trusting or logging it.

        Raises:
            InvalidSignatureError: signature mismatch or malformed signature.
        """
        if len(bundle.signed_prekey_signature) != _SIG_LEN:
            raise InvalidSignatureError(
                f'signed prekey signature must be {_SIG_LEN} bytes'
            )
        try:
            bundle.identity_key.verify(
                bundle.signed_prekey.encode(), bundle.signed_prekey_signature
            )
        except BadSignatureError as exc:
            raise InvalidSignatureError(
                'signed prekey signature does not match the advertised identity key'
            ) from exc

    # ------------------------------------------------------------------
    # Diffie-Hellman primitive
    # ------------------------------------------------------------------

    @staticmethod
    def _dh(private_key: PrivateKey, public_key: PublicKey) -> bytes:
        """
        One X25519 Diffie-Hellman operation.

        Raises:
            InvalidKeyError: degenerate (all-zero) result, which is what a
                low-order or otherwise non-generator public key produces.
        """
        try:
            shared = nacl.bindings.crypto_scalarmult(
                bytes(private_key), bytes(public_key)
            )
        except CryptoError as exc:
            raise InvalidKeyError('X25519 rejected the public key') from exc
        if shared == b'\x00' * _DH_LEN:
            raise InvalidKeyError(
                'degenerate X25519 output: public key is a low-order point'
            )
        return shared

    def _initiator_dh_parts(
        self,
        identity_private: SigningKey,
        ephemeral: PrivateKey,
        bundle: PreKeyBundle,
    ) -> Tuple[DhPart, ...]:
        """
        DH1 = DH(IK_A, SPK_B)
        DH2 = DH(EK_A, IK_B)
        DH3 = DH(EK_A, SPK_B)
        DH4 = DH(EK_A, OPK_B)   -- only when the bundle carries a one-time prekey
        PQ  = KEM(EK_A, PQK_B)  -- only when the bundle carries a PQ prekey

        The post-quantum ciphertext is returned alongside rather than kept on
        ``self``: a state object that carried a side effect would make
        :meth:`_initiator_dh_parts` unsafe to call twice, and the caller needs
        the ciphertext anyway.
        """
        identity_private_x = identity_private_x25519(identity_private)
        # The bundle's Ed25519 identity key is converted for the DH2 input. The
        # conversion is deterministic, so DH2 still commits to the signed key.
        identity_public_x = identity_public_x25519(bundle.identity_key)

        parts = [
            DhPart(LABEL_IK_SPK,
                   self._dh(identity_private_x, bundle.signed_prekey)),
            DhPart(LABEL_EK_IK,
                   self._dh(ephemeral, identity_public_x)),
            DhPart(LABEL_EK_SPK,
                   self._dh(ephemeral, bundle.signed_prekey)),
        ]
        if bundle.one_time_prekey is not None:
            parts.append(DhPart(
                LABEL_EK_OPK, self._dh(ephemeral, bundle.one_time_prekey)
            ))
        ciphertext = None
        if bundle.pq_prekey is not None:
            shared, ciphertext = pq_encapsulate(
                pq_load_public_key(bundle.pq_prekey.public_bytes_raw())
            )
            parts.append(DhPart(LABEL_KEM, shared))
        return tuple(parts), ciphertext

    def _responder_dh_parts(
        self,
        identity_private: SigningKey,
        signed_prekey_private: PrivateKey,
        one_time_prekey_private: Optional[PrivateKey],
        pq_prekey_private,
        pq_ciphertext: Optional[bytes],
        init: HandshakeInit,
    ) -> Tuple[DhPart, ...]:
        """
        DH1 = DH(SPK_B, IK_A)
        DH2 = DH(IK_B, EK_A)
        DH3 = DH(SPK_B, EK_A)
        DH4 = DH(OPK_B, EK_A)   -- only when a one-time prekey is present
        PQ  = Decaps(PQK_B, ct)  -- only when a PQ prekey is referenced
        """
        identity_private_x = identity_private_x25519(identity_private)

        parts = [
            DhPart(LABEL_IK_SPK,
                   self._dh(signed_prekey_private, init.identity_key_x25519)),
            DhPart(LABEL_EK_IK,
                   self._dh(identity_private_x, init.ephemeral_key)),
            DhPart(LABEL_EK_SPK,
                   self._dh(signed_prekey_private, init.ephemeral_key)),
        ]
        if init.one_time_prekey_id is not None:
            if one_time_prekey_private is None:
                raise InvalidHandshakeError(
                    'initiator referenced a one-time prekey but none was supplied'
                )
            parts.append(DhPart(
                LABEL_EK_OPK,
                self._dh(one_time_prekey_private, init.ephemeral_key),
            ))
        if init.pq_prekey_id is not None:
            if pq_prekey_private is None or pq_ciphertext is None:
                raise InvalidHandshakeError(
                    'initiator referenced a post-quantum prekey but none was '
                    'supplied'
                )
            parts.append(DhPart(
                LABEL_KEM, pq_decapsulate(pq_prekey_private, pq_ciphertext)
            ))
        return tuple(parts)

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def initiate_handshake(
        self,
        identity_private: SigningKey,
        bundle: PreKeyBundle,
        associated_data: bytes = b'',
        minimum_signed_prekey_id: Optional[int] = None,
    ) -> Tuple[X3DHState, HandshakeInit]:
        """
        Initiate a handshake as the initiator.

        The bundle's signed-prekey signature is verified *before* any bundle key
        is used, so a bundle whose signed prekey was substituted by an active
        attacker raises :class:`InvalidSignatureError` instead of silently
        producing a shared secret the attacker also knows.

        Args:
            minimum_signed_prekey_id: refuse a bundle whose ``signed_prekey_id``
                is older. A peer told that a rotation happened can require it,
                which is the only way to retire a cached bundle: its signature
                stays valid forever, because the identity key did not change.

        Returns:
            ``(state, init)``. ``init`` must be delivered to the responder and
            passed to :meth:`receive_handshake` unmodified; it carries the proof
            that the initiator owns the identity key it names.

        Raises:
            InvalidSignatureError: bundle authentication failed.
            RetiredSignedPrekey: the bundle is older than the required minimum.
        """
        if not isinstance(identity_private, SigningKey):
            raise InvalidHandshakeError('identity_private must be a SigningKey')
        if not isinstance(bundle, PreKeyBundle):
            raise InvalidHandshakeError('bundle must be a PreKeyBundle')

        # Authenticate the bundle first: nothing below this line may run on an
        # unverified bundle.
        self.verify_bundle(bundle)

        if minimum_signed_prekey_id is not None and \
                bundle.signed_prekey_id < minimum_signed_prekey_id:
            raise RetiredSignedPrekey(
                f'bundle signed_prekey_id {bundle.signed_prekey_id} is older '
                f'than the required {minimum_signed_prekey_id}'
            )

        ephemeral = PrivateKey.generate()
        dh_parts, pq_ciphertext = self._initiator_dh_parts(
            identity_private, ephemeral, bundle
        )
        identity_public = identity_private.verify_key

        transcript = _init_transcript(
            identity_public,
            identity_public_x25519(identity_public),
            ephemeral.public_key,
            bundle.one_time_prekey_id,
            pq_ciphertext,
            bundle.pq_prekey_id,
            associated_data,
        )
        signature = identity_private.sign(transcript).signature

        init = HandshakeInit(
            identity_key=identity_public,
            identity_key_x25519=identity_public_x25519(identity_public),
            ephemeral_key=ephemeral.public_key,
            signature=signature,
            one_time_prekey_id=bundle.one_time_prekey_id,
            pq_ciphertext=pq_ciphertext,
            pq_prekey_id=bundle.pq_prekey_id,
        )
        state = X3DHState(
            dh_parts=dh_parts,
            ephemeral_key=ephemeral,
            identity_key=identity_public,
            one_time_prekey_id=bundle.one_time_prekey_id,
            pq_prekey_id=bundle.pq_prekey_id,
            pq_ciphertext=pq_ciphertext,
        )
        return state, init

    def receive_handshake(
        self,
        identity_private: SigningKey,
        signed_prekey_private: PrivateKey,
        one_time_prekey_private: Optional[PrivateKey],
        init: HandshakeInit,
        associated_data: bytes = b'',
        pq_prekey_private=None,
    ) -> X3DHState:
        """
        Process an initiator handshake as the responder.

        The initiator's identity key is taken from ``init`` and authenticated
        against the signature carried in ``init``, rather than being accepted
        from the caller.

        Raises:
            InvalidSignatureError: the initiator did not prove ownership of the
                identity key it presented, or the transcript was tampered with.
        """
        if not isinstance(init, HandshakeInit):
            raise InvalidHandshakeError('init must be a HandshakeInit')

        init.verify(associated_data)

        # The DH2 input is re-derived from the verified identity key and
        # compared, so a mismatch cannot be smuggled past verification.
        expected_x = identity_public_x25519(init.identity_key)
        if not hmac.compare_digest(bytes(expected_x), bytes(init.identity_key_x25519)):
            raise InvalidSignatureError(
                'identity_key_x25519 does not match identity_key'
            )

        dh_parts = self._responder_dh_parts(
            identity_private, signed_prekey_private, one_time_prekey_private,
            pq_prekey_private, init.pq_ciphertext, init,
        )
        return X3DHState(
            dh_parts=dh_parts,
            ephemeral_key=None,
            identity_key=init.identity_key,
            one_time_prekey_id=init.one_time_prekey_id,
            pq_prekey_id=init.pq_prekey_id,
            pq_ciphertext=init.pq_ciphertext,
        )

    def begin(
        self,
        identity_private: SigningKey,
        bundle: PreKeyBundle,
        associated_data: bytes = b'',
        minimum_signed_prekey_id: Optional[int] = None,
    ) -> Tuple['X3DHSession', HandshakeInit]:
        """
        Initiate a handshake and get a session that is not yet usable.

        Same as :meth:`initiate_handshake`, but the returned session refuses to
        yield a root key until the responder's key confirmation verifies. Prefer
        this over :meth:`initiate_handshake` for anything that will send a
        message.
        """
        state, init = self.initiate_handshake(
            identity_private, bundle, associated_data, minimum_signed_prekey_id
        )
        session = X3DHSession(
            state,
            is_initiator=True,
            alice_identity=identity_private.verify_key,
            alice_ephemeral=init.ephemeral_key,
            bob_identity=bundle.identity_key,
            associated_data=associated_data,
        )
        return session, init

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
        """Thin delegate to the shared HKDF implementation."""
        return kdf_extract(salt, ikm)

    @staticmethod
    def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
        """Thin delegate to the shared HKDF implementation."""
        return kdf_expand(prk, info, length)

    @staticmethod
    def derive_master_key(
        dh_parts: Sequence[DhPart],
        context: bytes = DEFAULT_CONTEXT,
    ) -> bytes:
        """
        Derive the 32-byte root key from the individual contributions.

        The specification defines ``SK = KDF(F || DH1 || DH2 || DH3 || DH4)``
        with ``F = 0xFF * 32`` for X25519.

        This takes the *separate*, labelled contributions rather than one
        concatenated buffer. Two reasons, and the second one appeared only when
        post-quantum support was added:

        - The 3-DH and 4-DH cases are 96 and 128 bytes, and the already-prefixed
          3-DH form ``F || DH1 || DH2 || DH3`` is also 128 bytes, so a
          length-sniffing implementation cannot tell the last two apart and will
          silently derive different keys on the two sides. Taking a sequence
          removes that ambiguity.
        - Every contribution is 32 bytes, so with an ML-KEM secret in the mix a
          KDF handed a bare buffer cannot tell "three X25519 outputs plus a
          one-time prekey" from "three X25519 outputs plus an ML-KEM secret".
          The labels are mixed in here, so the two arrangements produce
          different keys instead of the same one.

        ``context`` is mixed in as the HKDF ``info``. The specification leaves
        ``info`` empty; a non-empty protocol-specific context is deliberate
        domain separation and does not weaken the construction. This is a
        documented deviation, and it also means these keys are not
        interchangeable with libsignal's.

        Raises:
            TypeError: ``dh_parts`` is a single buffer, or bare 32-byte values,
                not a sequence of :class:`DhPart`.
            ValueError: a part is malformed.
        """
        if isinstance(dh_parts, (bytes, bytearray, memoryview)):
            raise TypeError(
                'derive_master_key expects the separate DH outputs as a list or '
                'tuple, not one concatenated buffer: a buffer is ambiguous '
                'between the 3-DH and 4-DH cases'
            )
        parts = list(dh_parts)
        if not parts:
            raise ValueError('X3DH has at least three DH outputs, got 0')
        for part in parts:
            if not isinstance(part, DhPart):
                raise TypeError(
                    'derive_master_key expects DhPart values with labels, not '
                    f'bare 32-byte outputs: got {type(part).__name__}. A bare '
                    'output carries no indication of which key produced it, so '
                    'an ML-KEM contribution and an X25519 one would be '
                    'indistinguishable.'
                )
            if len(part.value) != _DH_LEN:
                raise ValueError(
                    f'DH output {part.label!r} must be {_DH_LEN} bytes, got '
                    f'{len(part.value)}'
                )
        labels = [part.label for part in parts]
        if len(set(labels)) != len(labels):
            # Two contributions with the same label means a caller assembled the
            # handshake wrong, and mixing it in would silently produce a key
            # neither specification nor this code has an opinion about.
            raise ValueError(f'duplicate DH part labels: {labels}')

        salt = b'\x00' * _DH_LEN
        prk = kdf_extract(salt, X25519_F + parts_value(parts))
        return kdf_expand(prk, context, _DH_LEN)


def parts_value(parts: Sequence[DhPart]) -> bytes:
    """
    Length-prefixed label-and-value concatenation for the KDF input.

    Public because it is what a transcript-equivalent computation needs, and
    because a second implementation of it on the other side of an HKDF is the
    exact failure :mod:`src.crypto.kdf` documents.
    """
    return b''.join(
        _u32(len(part.label)) + part.label + part.value for part in parts
    )


@dataclass
class X3DHResponder:
    """
    Responder side of X3DH, with single-use prekey enforcement.

    Holds the long-term identity key, the active signed prekey, an
    :class:`~src.crypto.prekey_store.PreKeyStore` and a
    :class:`~src.crypto.pq.PQPreKeyStore`. The stores' atomic ``consume`` is
    what makes a one-time prekey genuinely single use, and the post-quantum
    store is not optional: a bundle without an ML-KEM key produces a session
    whose forward secrecy will not survive a future break of X25519.

    Both pools are spent together and both are checked first. Handshakes
    arrive in any order, so the alternative -- consume as you go -- burns an
    X25519 prekey on a handshake that then fails for want of a post-quantum one,
    which is a denial of service against every future session.
    """

    identity_private: SigningKey
    signed_prekey_private: PrivateKey
    prekey_store: PreKeyStore
    pq_prekey_store: Optional[PQPreKeyStore] = None
    context: bytes = DEFAULT_CONTEXT
    signed_prekey_id: int = 0

    def __post_init__(self) -> None:
        self._x3dh = X3DH()
        # Publish exactly the prekey whose private half handle_init will use,
        # so the advertised signature and the DH key cannot drift apart.
        self._signed_prekey_public = self.signed_prekey_private.public_key
        self._signed_prekey_signature = self.identity_private.sign(
            self._signed_prekey_public.encode()
        ).signature

    @property
    def identity_public(self) -> VerifyKey:
        return self.identity_private.verify_key

    @property
    def is_post_quantum(self) -> bool:
        """Whether this responder can hand out post-quantum prekeys."""
        return self.pq_prekey_store is not None

    def publish_bundle(
        self,
        *,
        allow_no_one_time_prekey: bool = False,
        allow_no_pq: bool = False,
        reserve: bool = True,
    ) -> PreKeyBundle:
        """
        Build the bundle to hand to an initiator.

        Args:
            allow_no_one_time_prekey: permit returning a bundle with no
                one-time prekey. That handshake is the 3-DH variant, which works
                but has **no forward secrecy**, so it is opt-in rather than the
                default. A caller that accepts the downgrade is expected to
                record that it did.
            allow_no_pq: permit returning a bundle with no post-quantum prekey.
                The session then carries no ML-KEM contribution, so its forward
                secrecy ends the day X25519 is broken. Opt-in for the same
                reason: the caller has to be the one that decides a session
                without it is acceptable, because the peer cannot tell.
            reserve: take the prekeys out of circulation while they are
                advertised. On by default. Without it, two calls hand out the
                same prekey and the second handshake is refused by ``consume``,
                which is a denial of service against a legitimate peer caused by
                nothing more than a naive server. Use :meth:`release_bundle` to
                return advertised prekeys when the handshake never arrives.

        Raises:
            PrekeyPoolExhausted: the X25519 pool is empty and the caller did not
                opt in to the weaker variant.
            PQPreKeyExhausted: the post-quantum pool is empty, or this responder
                has no post-quantum store, and the caller did not opt in.

        A reserved prekey stays consumable, so the handshake it was advertised
        for still succeeds. What changes is that a *second* bundle cannot carry
        it.

        Both pools are checked before either is reserved. Reserving the X25519
        prekey and only then discovering the post-quantum pool is empty would
        leave that prekey held out of circulation for a bundle that was never
        published, and a client whose pool is low would lose one key per failed
        attempt until it had none left -- a self-inflicted outage, and one an
        attacker could trigger by simply letting the post-quantum pool run dry.
        """
        # Look before leaping: availability first, reservation second.
        candidate = self.prekey_store.peek_any()
        if candidate is None and not allow_no_one_time_prekey:
            raise PrekeyPoolExhausted(
                'no one-time prekeys available; replenish the pool, or pass '
                'allow_no_one_time_prekey=True to accept a session without '
                'forward secrecy'
            )

        if self.pq_prekey_store is not None:
            pq_available = self.pq_prekey_store.peek_any()
        else:
            pq_available = None
        if pq_available is None and not allow_no_pq:
            raise PQPreKeyExhausted(
                'no post-quantum one-time prekeys available. A session without '
                'one has no ML-KEM contribution, so its forward secrecy does '
                'not survive a future break of X25519. Replenish the pool, give '
                'this responder a PQPreKeyStore, or pass allow_no_pq=True to '
                'accept that downgrade explicitly.'
            )

        if reserve:
            candidate = self.prekey_store.reserve_next() if candidate else None
            if self.pq_prekey_store is not None and pq_available is not None:
                pq_candidate = self.pq_prekey_store.reserve_next()
            else:
                pq_candidate = None
        else:
            candidate, pq_candidate = candidate, pq_available

        return PreKeyBundle(
            identity_key=self.identity_public,
            signed_prekey=self._signed_prekey_public,
            signed_prekey_signature=self._signed_prekey_signature,
            one_time_prekey=None if candidate is None else candidate.public_key,
            one_time_prekey_id=None if candidate is None else candidate.key_id,
            signed_prekey_id=self.signed_prekey_id,
            pq_prekey=None if pq_candidate is None else pq_candidate.public_key,
            pq_prekey_id=None if pq_candidate is None else pq_candidate.key_id,
        )

    def release_bundle(self, bundle: PreKeyBundle) -> None:
        """
        Return the prekeys an advertised bundle carried to circulation.

        For the case where the handshake never arrived. Once ``handle_init`` has
        run a key is consumed and this raises, which is the intended way to
        notice the mistake.

        Raises:
            PreKeyStoreError: an X25519 prekey was already consumed.
            PQError: a post-quantum prekey was already consumed.
        """
        if bundle.one_time_prekey_id is not None:
            self.prekey_store.release(bundle.one_time_prekey_id)
        if bundle.pq_prekey_id is not None:
            if self.pq_prekey_store is None:
                raise PQError(
                    'this bundle carried a post-quantum prekey but the '
                    'responder has no post-quantum store'
                )
            self.pq_prekey_store.release(bundle.pq_prekey_id)

    def _spend_prekeys(self, init: HandshakeInit) -> Tuple[Optional[OneTimePreKey],
                                                           Optional[object]]:
        """
        Take the prekeys ``init`` references, in one step.

        Both pools are checked before either is touched, so a handshake that
        names an unavailable prekey costs nothing. Authentication has already
        happened by the time this runs, which is the other half of why that
        order matters: an unauthenticated handshake must not be able to spend
        anything at all.

        Raises:
            NoSuchPreKey, OneTimePreKeyAlreadyUsed: the X25519 prekey.
            NoSuchPQPreKey, PQPreKeyAlreadyUsed: the post-quantum prekey.
        """
        if init.one_time_prekey_id is not None:
            one_time = self.prekey_store.peek(init.one_time_prekey_id)
        else:
            one_time = None

        pq_private = None
        if init.pq_prekey_id is not None:
            if self.pq_prekey_store is None:
                raise NoSuchPQPreKey(
                    'the initiator referenced a post-quantum prekey and this '
                    'responder has no post-quantum store; the handshake cannot '
                    'be answered with a key both sides share'
                )
            pq_candidate = self.pq_prekey_store.peek(init.pq_prekey_id)
            pq_private = pq_candidate.private_key

        # Checked, now spent.
        if one_time is not None:
            one_time = self.prekey_store.consume(init.one_time_prekey_id)
        if pq_private is not None:
            pq_private = self.pq_prekey_store.consume(
                init.pq_prekey_id
            ).private_key
        return one_time, pq_private

    def respond(
        self,
        init: HandshakeInit,
        associated_data: bytes = b'',
    ) -> 'X3DHSession':
        """
        Authenticate an initiator handshake and get a responder session.

        Equivalent to :meth:`handle_init`, but returns a session that can
        produce a key confirmation, which is what lets the initiator verify
        that this side derived the same root key.
        """
        if not isinstance(init, HandshakeInit):
            raise InvalidHandshakeError('init must be a HandshakeInit')

        # Authenticate BEFORE spending a prekey. See handle_init.
        init.verify(associated_data)

        one_time, pq_private = self._spend_prekeys(init)

        state = self._x3dh.receive_handshake(
            self.identity_private,
            self.signed_prekey_private,
            None if one_time is None else one_time.private_key,
            init,
            associated_data,
            pq_prekey_private=pq_private,
        )
        return X3DHSession(
            state,
            is_initiator=False,
            alice_identity=init.identity_key,
            alice_ephemeral=init.ephemeral_key,
            bob_identity=self.identity_private.verify_key,
            associated_data=associated_data,
        )

    def handle_init(
        self,
        init: HandshakeInit,
        associated_data: bytes = b'',
    ) -> X3DHState:
        """
        Verify and process an initiator handshake, consuming the one-time prekeys.

        The prekeys are consumed atomically before the shared secret is computed,
        so a replayed or duplicated handshake cannot reuse them.

        If a referenced prekey is already spent, the handshake is aborted. The
        initiator's signature covers both prekey ids, so the responder cannot
        silently drop the post-quantum contribution: the initiator would derive a
        different root key. Recovery is a fresh handshake against a fresh
        bundle.

        Raises:
            OneTimePreKeyAlreadyUsed: the referenced X25519 prekey was already
                spent. Accepting it would destroy forward secrecy.
            PQPreKeyAlreadyUsed: the referenced post-quantum prekey was already
                spent, for the same reason.
            NoSuchPreKey, NoSuchPQPreKey: a referenced prekey is unknown.
            InvalidSignatureError: the initiator failed authentication.
        """
        if not isinstance(init, HandshakeInit):
            raise InvalidHandshakeError('init must be a HandshakeInit')

        # Authenticate BEFORE spending a prekey.
        #
        # Consuming first lets anyone burn the whole pool by sending forged
        # handshakes: each one would take a prekey and only then fail
        # verification, leaving the responder unable to accept genuine
        # handshakes with forward secrecy. The same applies to the
        # post-quantum pool, which is the one an attacker would most want to
        # drain, since an empty pool means every later session has to be
        # accepted without post-quantum protection or not at all.
        init.verify(associated_data)

        one_time, pq_private = self._spend_prekeys(init)

        return self._x3dh.receive_handshake(
            self.identity_private,
            self.signed_prekey_private,
            None if one_time is None else one_time.private_key,
            init,
            associated_data,
            pq_prekey_private=pq_private,
        )
