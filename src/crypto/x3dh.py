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


def _length_prefixed(*chunks: Optional[bytes]) -> bytes:
    """
    Concatenate chunks with explicit length prefixes.

    Length prefixing keeps the encoding unambiguous, so no two different field
    sets can produce the same byte string. Without it a signature could be
    replayed against a differently split set of fields.
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
    associated_data: bytes,
) -> bytes:
    """
    Canonical transcript covered by the initiator's handshake signature.

    Binds the identity key, both of its curve representations, the ephemeral
    key, the referenced prekey id and any external associated data.
    """
    opk = (None if one_time_prekey_id is None
           else one_time_prekey_id.to_bytes(8, 'big'))
    return _length_prefixed(
        INIT_TRANSCRIPT_LABEL,
        bytes(identity_key),
        bytes(identity_key_x25519),
        bytes(ephemeral_key),
        opk,
        associated_data,
    )


@dataclass(frozen=True)
class PreKeyBundle:
    """
    The prekey bundle published by a responder.

    ``signed_prekey_signature`` must be an Ed25519 signature over the raw
    32-byte ``signed_prekey``, as produced by
    :meth:`X3DH.generate_signed_prekey`.
    """

    identity_key: VerifyKey
    signed_prekey: PublicKey
    signed_prekey_signature: bytes
    one_time_prekey: Optional[PublicKey] = None
    one_time_prekey_id: Optional[int] = None

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


@dataclass(frozen=True)
class HandshakeInit:
    """
    The initiator's handshake message, carrying its own proof of possession.

    ``signature`` is an Ed25519 signature by ``identity_key`` over
    :meth:`transcript`. This is the only structure :meth:`X3DH.receive_handshake`
    accepts, so a handshake cannot be processed without proof that the sender
    owns the identity key it names.
    """

    identity_key: VerifyKey
    identity_key_x25519: PublicKey
    ephemeral_key: PublicKey
    signature: bytes
    one_time_prekey_id: Optional[int] = None

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

    def transcript(self, associated_data: bytes = b'') -> bytes:
        """Canonical, length-prefixed byte string covered by ``signature``."""
        return _init_transcript(
            self.identity_key,
            self.identity_key_x25519,
            self.ephemeral_key,
            self.one_time_prekey_id,
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


@dataclass
class X3DHState:
    """
    One side's handshake result.

    The individual DH outputs are kept rather than only their concatenation, so
    the KDF step cannot be handed an ambiguous buffer. Use :meth:`master_key` to
    derive the 32-byte root key for the Double Ratchet.
    """

    dh_parts: Tuple[bytes, ...]
    ephemeral_key: Optional[PrivateKey] = None
    identity_key: Optional[VerifyKey] = None
    one_time_prekey_id: Optional[int] = None

    def master_key(self, context: bytes = DEFAULT_CONTEXT) -> bytes:
        return X3DH.derive_master_key(self.dh_parts, context)

    @property
    def shared_secret(self) -> bytes:
        """
        Raw concatenation of the DH outputs, without F and without the KDF.

        Kept for tests and debugging. Never use it as a message or root key;
        use :meth:`master_key`.
        """
        return b''.join(self.dh_parts)


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
    transcript = X25519_F + _length_prefixed(
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
    def one_time_prekey_id(self) -> Optional[int]:
        return self._state.one_time_prekey_id

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
    ) -> Tuple[bytes, ...]:
        """
        DH1 = DH(IK_A, SPK_B)
        DH2 = DH(EK_A, IK_B)
        DH3 = DH(EK_A, SPK_B)
        DH4 = DH(EK_A, OPK_B)   -- only when the bundle carries a one-time prekey
        """
        identity_private_x = identity_private_x25519(identity_private)
        # The bundle's Ed25519 identity key is converted for the DH2 input. The
        # conversion is deterministic, so DH2 still commits to the signed key.
        identity_public_x = identity_public_x25519(bundle.identity_key)

        parts = [
            self._dh(identity_private_x, bundle.signed_prekey),
            self._dh(ephemeral, identity_public_x),
            self._dh(ephemeral, bundle.signed_prekey),
        ]
        if bundle.one_time_prekey is not None:
            parts.append(self._dh(ephemeral, bundle.one_time_prekey))
        return tuple(parts)

    def _responder_dh_parts(
        self,
        identity_private: SigningKey,
        signed_prekey_private: PrivateKey,
        one_time_prekey_private: Optional[PrivateKey],
        init: HandshakeInit,
    ) -> Tuple[bytes, ...]:
        """
        DH1 = DH(SPK_B, IK_A)
        DH2 = DH(IK_B, EK_A)
        DH3 = DH(SPK_B, EK_A)
        DH4 = DH(OPK_B, EK_A)   -- only when a one-time prekey is present
        """
        identity_private_x = identity_private_x25519(identity_private)

        parts = [
            self._dh(signed_prekey_private, init.identity_key_x25519),
            self._dh(identity_private_x, init.ephemeral_key),
            self._dh(signed_prekey_private, init.ephemeral_key),
        ]
        if init.one_time_prekey_id is not None:
            if one_time_prekey_private is None:
                raise InvalidHandshakeError(
                    'initiator referenced a one-time prekey but none was supplied'
                )
            parts.append(self._dh(one_time_prekey_private, init.ephemeral_key))
        return tuple(parts)

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def initiate_handshake(
        self,
        identity_private: SigningKey,
        bundle: PreKeyBundle,
        associated_data: bytes = b'',
    ) -> Tuple[X3DHState, HandshakeInit]:
        """
        Initiate a handshake as the initiator.

        The bundle's signed-prekey signature is verified *before* any bundle key
        is used, so a bundle whose signed prekey was substituted by an active
        attacker raises :class:`InvalidSignatureError` instead of silently
        producing a shared secret the attacker also knows.

        Returns:
            ``(state, init)``. ``init`` must be delivered to the responder and
            passed to :meth:`receive_handshake` unmodified; it carries the proof
            that the initiator owns the identity key it names.
        """
        if not isinstance(identity_private, SigningKey):
            raise InvalidHandshakeError('identity_private must be a SigningKey')
        if not isinstance(bundle, PreKeyBundle):
            raise InvalidHandshakeError('bundle must be a PreKeyBundle')

        # Authenticate the bundle first: nothing below this line may run on an
        # unverified bundle.
        self.verify_bundle(bundle)

        ephemeral = PrivateKey.generate()
        dh_parts = self._initiator_dh_parts(identity_private, ephemeral, bundle)
        identity_public = identity_private.verify_key

        transcript = _init_transcript(
            identity_public,
            identity_public_x25519(identity_public),
            ephemeral.public_key,
            bundle.one_time_prekey_id,
            associated_data,
        )
        signature = identity_private.sign(transcript).signature

        init = HandshakeInit(
            identity_key=identity_public,
            identity_key_x25519=identity_public_x25519(identity_public),
            ephemeral_key=ephemeral.public_key,
            signature=signature,
            one_time_prekey_id=bundle.one_time_prekey_id,
        )
        state = X3DHState(
            dh_parts=dh_parts,
            ephemeral_key=ephemeral,
            identity_key=identity_public,
            one_time_prekey_id=bundle.one_time_prekey_id,
        )
        return state, init

    def receive_handshake(
        self,
        identity_private: SigningKey,
        signed_prekey_private: PrivateKey,
        one_time_prekey_private: Optional[PrivateKey],
        init: HandshakeInit,
        associated_data: bytes = b'',
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
            identity_private, signed_prekey_private, one_time_prekey_private, init
        )
        return X3DHState(
            dh_parts=dh_parts,
            ephemeral_key=None,
            identity_key=init.identity_key,
            one_time_prekey_id=init.one_time_prekey_id,
        )

    def begin(
        self,
        identity_private: SigningKey,
        bundle: PreKeyBundle,
        associated_data: bytes = b'',
    ) -> Tuple['X3DHSession', HandshakeInit]:
        """
        Initiate a handshake and get a session that is not yet usable.

        Same as :meth:`initiate_handshake`, but the returned session refuses to
        yield a root key until the responder's key confirmation verifies. Prefer
        this over :meth:`initiate_handshake` for anything that will send a
        message.
        """
        state, init = self.initiate_handshake(identity_private, bundle, associated_data)
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
        dh_parts: Sequence[bytes],
        context: bytes = DEFAULT_CONTEXT,
    ) -> bytes:
        """
        Derive the 32-byte root key from the individual DH outputs.

        The specification defines ``SK = KDF(F || DH1 || DH2 || DH3 || DH4)``
        with ``F = 0xFF * 32`` for X25519.

        This takes the *separate* DH outputs rather than one concatenated
        buffer on purpose. The 3-DH and 4-DH cases are 96 and 128 bytes, and the
        already-prefixed 3-DH form ``F || DH1 || DH2 || DH3`` is also 128 bytes,
        so a length-sniffing implementation cannot tell the last two apart and
        will silently derive different keys on the two sides. Taking a sequence
        removes the ambiguity.

        ``context`` is mixed in as the HKDF ``info``. The specification leaves
        ``info`` empty; a non-empty protocol-specific context is deliberate
        domain separation and does not weaken the construction. This is a
        documented deviation, and it also means these keys are not
        interchangeable with libsignal's.

        Raises:
            TypeError: ``dh_parts`` is a single buffer, not a sequence of parts.
            ValueError: wrong number of parts, or a part that is not 32 bytes.
        """
        if isinstance(dh_parts, (bytes, bytearray, memoryview)):
            raise TypeError(
                'derive_master_key expects the separate DH outputs as a list or '
                'tuple, not one concatenated buffer: a buffer is ambiguous '
                'between the 3-DH and 4-DH cases'
            )
        parts = list(dh_parts)
        if len(parts) not in (3, 4):
            raise ValueError(f'X3DH uses 3 or 4 DH outputs, got {len(parts)}')
        for index, part in enumerate(parts):
            if len(part) != _DH_LEN:
                raise ValueError(
                    f'DH output {index} must be {_DH_LEN} bytes, got {len(part)}'
                )

        salt = b'\x00' * _DH_LEN
        prk = kdf_extract(salt, X25519_F + b''.join(parts))
        return kdf_expand(prk, context, _DH_LEN)


@dataclass
class X3DHResponder:
    """
    Responder side of X3DH, with single-use prekey enforcement.

    Holds the long-term identity key, the active signed prekey and a
    :class:`~src.crypto.prekey_store.PreKeyStore`. The store's atomic
    ``consume`` is what makes a one-time prekey genuinely single use.
    """

    identity_private: SigningKey
    signed_prekey_private: PrivateKey
    prekey_store: PreKeyStore
    context: bytes = DEFAULT_CONTEXT

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

    def publish_bundle(self) -> PreKeyBundle:
        """
        Build the bundle to hand to an initiator.

        Publication does not consume a one-time prekey: the handshake may never
        arrive, and consuming here would silently shrink the pool. The atomic
        decision happens in :meth:`handle_init`.

        A production server must additionally *reserve* the prekey it hands out,
        so two clients are never issued the same one. This reference
        implementation does not model reservation; see the module docstring of
        :mod:`src.crypto.prekey_store`.
        """
        candidate = self.prekey_store.peek_any()
        return PreKeyBundle(
            identity_key=self.identity_public,
            signed_prekey=self._signed_prekey_public,
            signed_prekey_signature=self._signed_prekey_signature,
            one_time_prekey=None if candidate is None else candidate.public_key,
            one_time_prekey_id=None if candidate is None else candidate.key_id,
        )

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

        # Authenticate BEFORE spending a prekey.
        #
        # Consuming first lets anyone drain the pool with forged handshakes.
        init.verify(associated_data)

        one_time_private: Optional[PrivateKey] = None
        if init.one_time_prekey_id is not None:
            one_time_private = self.prekey_store.consume(
                init.one_time_prekey_id
            ).private_key

        state = self._x3dh.receive_handshake(
            self.identity_private,
            self.signed_prekey_private,
            one_time_private,
            init,
            associated_data,
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
        Verify and process an initiator handshake, consuming the one-time prekey.

        The prekey is consumed atomically before the shared secret is computed,
        so a replayed or duplicated handshake cannot reuse it.

        If the referenced prekey is already spent, the handshake is aborted. The
        initiator's signature covers the prekey id, so the responder cannot
        silently downgrade to the 3-DH variant: the initiator would derive a
        different root key. Recovery is a fresh handshake against a fresh
        bundle.

        Raises:
            OneTimePreKeyAlreadyUsed: the referenced prekey was already spent.
                Accepting it would destroy forward secrecy.
            NoSuchPreKey: the referenced prekey id is unknown.
            InvalidSignatureError: the initiator failed authentication.
        """
        if not isinstance(init, HandshakeInit):
            raise InvalidHandshakeError('init must be a HandshakeInit')

        # Authenticate BEFORE spending a prekey.
        #
        # Consuming first lets anyone burn the whole pool by sending forged
        # handshakes: each one would take a prekey and only then fail
        # verification, leaving the responder unable to accept genuine
        # handshakes with forward secrecy.
        init.verify(associated_data)

        one_time_private: Optional[PrivateKey] = None
        if init.one_time_prekey_id is not None:
            # Raises if the prekey was already used; that is a security event,
            # not a recoverable condition, so it propagates.
            one_time_private = self.prekey_store.consume(
                init.one_time_prekey_id
            ).private_key

        return self._x3dh.receive_handshake(
            self.identity_private,
            self.signed_prekey_private,
            one_time_private,
            init,
            associated_data,
        )
