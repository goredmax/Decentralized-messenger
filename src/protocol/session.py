"""
A one-to-one conversation: X3DH handshake, root key, Double Ratchet, frames.

Why this module exists
----------------------
The cryptographic core was written and tested as if the two people talking had
never met. No module in the repository -- not ``src/crypto``, not
``src/protocol``, not ``src/storage`` -- called ``DoubleRatchet(...)``. All 460
tests assembled a ratchet by hand around a root key such as ``b'\\x01' * 32``,
which is tied to no handshake at all. Both sides derived the same thing from
it, and that agreement between them is exactly the case in which the suite
stays green while nothing connects the two halves of the protocol.

This module is where the halves meet. It holds no cryptography: it decides
*which* keys become the first ratchet keys, and moves messages into bytes
through :mod:`src.protocol.wire`.

Which keys become the first ratchet keys
----------------------------------------
The Signal specification requires the initiator's first ratchet key to be its
own X25519 key and the remote one to be the responder's signed prekey:

* initiator: ``dh_local = X25519(own Ed25519 identity key)``,
  ``dh_remote = the bundle's signed prekey``, ratcheting immediately;
* responder: ``dh_local = own signed prekey``, ``dh_remote = None``, waiting
  for the first header to learn the initiator's key.

That choice is not free: the signed prekey's signature covers exactly this
key, so the verification in :meth:`X3DH.initiate_handshake` simultaneously
proves the responder honestly published the key the ratchet will use. Wiring
the ratchet to any other key would break that link.

The consequence is worth stating: the responder's first ratchet key is the
signed prekey it publishes, so it is not unique to one session. Two sessions
started from one bundle use it as the initial ratchet key. That is
acceptable -- their root keys differ, and both sides move to fresh keys after
the first DH step -- but it is why a valid signature on a bundle must not be
read as proof that a session is unique.

Associated data
---------------
Every message carries associated data, and the caller does not supply it. It
is :func:`session_associated_data`: a version label, both identity keys in a
canonical order (initiator first, responder second) and the session protocol
version. Both sides know both keys, so the values match; a frame from another
session does not, so moving one breaks the AEAD instead of passing as the same
conversation. The specification leaves these bytes to the application, and
leaving them empty throws away a property that costs nothing.

Note what the associated data does with the role: nothing directly. Both sides
must derive the same bytes, so no field may differ by role. But the *order* of
the two identity keys is the role, both sides agree on that order because each
knows who started, and :meth:`Session.from_state` reorders the keys it is given
according to the role the caller claims. So one wrong argument is caught at the
AEAD, and two compensating ones are not; that method says which.

Whose turn it is
----------------
Both sides can send and both can receive. A session does **not** decide that
by hand, and ``send``/``receive`` do not mean "I am the sender": ``send``
encrypts into the current sending chain, ``receive`` decrypts from the
current receiving chain, and the Double Ratchet decides for itself who is
performing a DH ratchet. Getting this wrong costs a desynchronised session,
not an error message.
"""

from typing import Callable, Optional, Tuple

from nacl.public import PrivateKey
from nacl.signing import SigningKey, VerifyKey

from ..crypto.double_ratchet import (
    PROTOCOL_VERSION,
    DoubleRatchet,
    SessionState,
)
from ..crypto.key_management import identity_private_x25519
from ..crypto.x3dh import (
    PreKeyBundle,
    X3DH,
    X3DHSession,
    X3DHResponder,
    length_prefixed,
)
from .padding import DEFAULT_PROFILE, PaddingProfile, pad, unpad
from .wire import (
    decode_handshake_init,
    decode_key_confirmation,
    decode_ratchet_message,
    encode_handshake_init,
    encode_key_confirmation,
    encode_ratchet_message,
)

#: Version label for the associated data. Bumped with the format rather than
#: with the implementation, so the bump happens here and not in every caller.
SESSION_AD_LABEL = b"DecentralizedMessenger-Session-v1"


class SessionError(Exception):
    """Base class for session-level failures."""


def session_associated_data(
    initiator_identity: VerifyKey,
    responder_identity: VerifyKey,
    protocol_version: int = PROTOCOL_VERSION,
) -> bytes:
    """
    Build the AEAD associated data shared by both sides of a session.

    Args:
        initiator_identity: identity key of whoever started the handshake.
        responder_identity: identity key of whoever answered it.
        protocol_version: the session protocol version, so a frame produced
            under an older key schedule cannot be replayed into this one.

    Returns:
        The associated data. Both sides must compute the same value, so the
        order of the two identities is fixed rather than chosen per call.
    """
    if not isinstance(initiator_identity, VerifyKey):
        raise SessionError('initiator_identity must be a VerifyKey')
    if not isinstance(responder_identity, VerifyKey):
        raise SessionError('responder_identity must be a VerifyKey')
    return length_prefixed(
        SESSION_AD_LABEL,
        bytes(initiator_identity),
        bytes(responder_identity),
        protocol_version.to_bytes(4, 'big'),
    )


class Session:
    """
    One side of a one-to-one conversation, from handshake to messages.

    Built by :func:`initiate` or :func:`accept`; use :meth:`from_state` to
    resume a persisted one. The Double Ratchet is created lazily, on first
    use, because the initiator cannot have one before the responder's
    confirmation verifies -- the root key it needs stays locked until then.
    Deferring the construction is what makes that lock mean something:
    :attr:`ratchet` propagates
    :class:`~src.crypto.x3dh.KeyConfirmationRequired` rather than building a
    ratchet over a root key nobody has proved they share.
    """

    def __init__(
        self,
        *,
        is_initiator: bool,
        ratchet: Optional[DoubleRatchet] = None,
        ratchet_factory: Optional[Callable[[], DoubleRatchet]] = None,
        x3dh_session: Optional[X3DHSession] = None,
        initiator_identity: VerifyKey,
        responder_identity: VerifyKey,
    ) -> None:
        if (ratchet is None) == (ratchet_factory is None):
            raise SessionError(
                'a Session needs either a ratchet or a factory that builds one, '
                'not both and not neither'
            )
        if x3dh_session is not None and \
                x3dh_session.is_initiator != is_initiator:
            raise SessionError(
                'is_initiator contradicts the X3DH session it was given'
            )
        self._ratchet = ratchet
        self._ratchet_factory = ratchet_factory
        self._x3dh = x3dh_session
        self._is_initiator = is_initiator
        self._initiator_identity = initiator_identity
        self._responder_identity = responder_identity
        self._ad = session_associated_data(
            initiator_identity, responder_identity
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initiator(self) -> bool:
        """Whether this side started the handshake."""
        return self._is_initiator

    @property
    def is_confirmed(self) -> bool:
        """
        Whether the peer proved it derived the same root key.

        True from the start for the responder: it authenticated the initiator
        and computed the secret itself, so there is nothing left to check. The
        initiator starts unconfirmed.
        """
        if self._x3dh is None:
            # A restored session was confirmed before it was ever written down.
            return True
        return self._x3dh.is_confirmed

    @property
    def local_identity(self) -> VerifyKey:
        return (
            self._initiator_identity
            if self._is_initiator
            else self._responder_identity
        )

    @property
    def peer_identity(self) -> VerifyKey:
        return (
            self._responder_identity
            if self._is_initiator
            else self._initiator_identity
        )

    @property
    def one_time_prekey_id(self) -> Optional[int]:
        """Which one-time prekey this handshake spent, if any."""
        return self._x3dh.one_time_prekey_id if self._x3dh is not None else None

    @property
    def pq_prekey_id(self) -> Optional[int]:
        """Which post-quantum prekey this handshake spent, if any."""
        return self._x3dh.pq_prekey_id if self._x3dh is not None else None

    @property
    def is_post_quantum(self) -> bool:
        """
        Whether the root key this session runs on includes an ML-KEM
        contribution.

        False means the session's forward secrecy ends the day X25519 is broken.
        Worth reporting rather than assuming, and worth recording for a session
        that was accepted on an explicit downgrade.
        """
        return self._x3dh.is_post_quantum if self._x3dh is not None else False

    @property
    def ratchet(self) -> DoubleRatchet:
        """
        The live Double Ratchet, created on first access.

        Raises:
            KeyConfirmationRequired: the initiator has not verified the
                responder's confirmation, so there is no root key to build over.
        """
        if self._ratchet is None:
            self._ratchet = self._ratchet_factory()
            self._ratchet_factory = None
        return self._ratchet

    @property
    def state(self) -> SessionState:
        """Serialisable state, for :class:`~src.storage.store.SessionStore`."""
        return self.ratchet.state

    @property
    def associated_data(self) -> bytes:
        """The AEAD associated data this session sends and expects."""
        return self._ad

    # ------------------------------------------------------------------
    # Handshake completion
    # ------------------------------------------------------------------

    def confirm(self, frame: bytes) -> None:
        """
        Initiator side: verify the responder's key confirmation.

        Args:
            frame: a ``KIND_KEY_CONFIRMATION`` frame, as returned by
                :func:`accept`.

        Raises:
            SessionError: called on the responder, which has nothing to verify.
            KeyConfirmationFailed: the MAC does not match, so the two sides
                derived different root keys. The session is left unconfirmed
                and unusable, and the cure is a new handshake rather than a
                retry.
            WireError: the frame is malformed.
        """
        if self._x3dh is None:
            raise SessionError('this session is already established')
        if not self._is_initiator:
            raise SessionError(
                'only the initiator verifies a key confirmation; the responder '
                'produces it'
            )
        if self._x3dh.is_confirmed:
            # Verifying the same MAC twice would succeed and prove nothing.
            # More importantly, it would let a caller treat a confirmed
            # session as one still waiting on the peer.
            raise SessionError('this session is already confirmed')
        self._x3dh.verify_key_confirmation(decode_key_confirmation(frame))

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def send(self, plaintext: bytes,
             profile: Optional[PaddingProfile] = None) -> bytes:
        """
        Encrypt one message into a complete wire frame, padded.

        Two layers of length hiding, both from one profile: the plaintext is
        padded under the AEAD so the message length is hidden from anyone who
        can decrypt but should not learn it, and the frame is quantised in
        :mod:`src.protocol.wire` so its length on the path is a bucket rather
        than a measurement. See :mod:`src.protocol.padding` for what this does
        not hide, which is most of it.

        The responder cannot send first. Its ratchet starts without a remote DH
        key because the specification has it learn the initiator's ratchet key
        from the first header it receives; encrypting before that arrives
        raises :class:`~src.crypto.double_ratchet.InvalidSessionStateError`
        rather than guessing a remote key.

        Args:
            plaintext: the message.
            profile: the size ladder, or ``None`` for
                :data:`~src.protocol.padding.DEFAULT_PROFILE`. A profile with a
                floor of zero produces unpadded frames, which is only useful
                for measuring what padding is worth.

        Raises:
            KeyConfirmationRequired: the handshake is not confirmed yet.
            InvalidSessionStateError: this side has no remote ratchet key yet.
            PaddingError: the message does not fit in a frame once padded.
            FrameTooLarge: the frame would exceed :data:`MAX_FRAME_BYTES`.
        """
        if not isinstance(plaintext, (bytes, bytearray, memoryview)):
            raise SessionError('plaintext must be bytes')
        if profile is None:
            profile = DEFAULT_PROFILE
        padded = pad(bytes(plaintext), profile)
        ciphertext, header = self.ratchet.encrypt(padded, self._ad)
        return encode_ratchet_message(ciphertext, header, profile)

    def receive(self, frame: bytes,
                profile: Optional[PaddingProfile] = None) -> bytes:
        """
        Decrypt one complete wire frame and strip its padding.

        Unpadding happens after the AEAD, so a padding failure here means the
        padding itself was wrong -- a bug or a downgrade -- not a corrupted
        packet, which the AEAD would have rejected already.

        Args:
            frame: the frame.
            profile: the size ladder, or ``None`` for the default. Only the
                plaintext bucket is consulted on the way in, so a peer with a
                different profile still decodes; the frame's own quantisation is
                verified by neither side.

        Raises:
            WireError: the frame is malformed or of an unknown kind.
            PaddingError: the decrypted plaintext is not a padded message.
            DoubleRatchetError: the frame did not authenticate under this
                session, which is what a replay from another session, a forged
                header and a tampered ciphertext all look like from here.
        """
        message = decode_ratchet_message(frame)
        padded = self.ratchet.decrypt(
            message.ciphertext, message.to_header(), self._ad
        )
        return unpad(padded)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_state(
        cls,
        state,
        *,
        is_initiator: bool,
        local_identity: VerifyKey,
        peer_identity: VerifyKey,
    ) -> 'Session':
        """
        Resume a session from a persisted :class:`SessionState`.

        Args:
            state: a ``SessionState`` or a serialized dict, as returned by
                :meth:`~src.storage.store.SessionStore.load`.
            is_initiator: which side this peer was on. The state does not record
                it, and the role is not stored in the associated data either --
                but it does fix the *order* of the two identity keys there, so a
                wrong role passed with a correctly ordered pair of identities
                changes the associated data and every message fails loudly. Two
                wrong arguments cancel out: a wrong role together with the two
                identities handed over the wrong way round. The caller states
                the role, and the CLI takes it from the command line, which is
                where it comes from in the first place.
            local_identity: this side's identity key.
            peer_identity: the other side's identity key.

        Raises:
            SessionError: the state cannot belong to the claimed role, see
                :meth:`_check_role_against_state`.
            InvalidSessionStateError: the state is malformed, or was written by
                a different protocol version.
        """
        _check_role_against_state(state, is_initiator)
        initiator, responder = (
            (local_identity, peer_identity)
            if is_initiator
            else (peer_identity, local_identity)
        )
        return cls(
            ratchet=DoubleRatchet.from_state(state),
            is_initiator=is_initiator,
            initiator_identity=initiator,
            responder_identity=responder,
        )


def _check_role_against_state(state, is_initiator: bool) -> None:
    """
    Reject a state that cannot belong to the role the caller claims.

    One direction is provable and is checked. An initiator ratchets the moment
    it is built, because the specification gives it the responder's signed
    prekey as its remote ratchet key, so a state carrying no remote ratchet key
    at all was written by a responder that had neither sent nor received. That
    is a contradiction rather than an ambiguity.

    The other direction is not checkable and is not pretended to be. Once a
    session has exchanged anything, both roles have a remote ratchet key and a
    sending chain, and the two states are the same shape. The asymmetry is
    stated in :meth:`Session.from_state` instead: this argument is the caller's
    to get right, and there is no state field that settles it.
    """
    document = state if isinstance(state, dict) else state.serialize()
    if is_initiator and document.get('dh_remote_pub') is None:
        raise SessionError(
            'this state has no remote ratchet key, so it was written by a '
            'responder that had not yet received a message; it cannot be the '
            "initiator's state"
        )


def _initiator_ratchet(
    identity_private: SigningKey,
    signed_prekey: bytes,
    root_key: bytes,
) -> DoubleRatchet:
    """
    Build the initiator's ratchet, per the specification.

    The first ratchet key is the initiator's own X25519 identity key and the
    remote one is the signed prekey, so the first DH output is one the bundle's
    signature already vouches for.
    """
    return DoubleRatchet(
        identity_private_x25519(identity_private),
        signed_prekey,
        root_key,
        is_initiator=True,
    )


def _responder_ratchet(
    signed_prekey_private: PrivateKey, root_key: bytes
) -> DoubleRatchet:
    """
    Build the responder's ratchet.

    ``remote_dh_public`` stays None: the responder learns the initiator's
    ratchet key from the first header it receives, which is how the
    specification avoids requiring it out of band.
    """
    return DoubleRatchet(
        signed_prekey_private,
        None,
        root_key,
        is_initiator=False,
    )


def initiate(
    identity_private: SigningKey,
    bundle: PreKeyBundle,
    *,
    x3dh: Optional[X3DH] = None,
    associated_data: bytes = b'',
    minimum_signed_prekey_id: Optional[int] = None,
) -> Tuple['Session', bytes]:
    """
    Start a session as the initiator.

    Args:
        identity_private: this side's Ed25519 identity key.
        bundle: the responder's prekey bundle, already fetched from wherever.
            Its signature is verified before any of its keys is used.
        associated_data: external context mixed into the handshake transcript
            and the key confirmation MAC. Independent of the message-level
            associated data, which this module derives itself.
        minimum_signed_prekey_id: refuse a bundle older than this, which is the
            only way to retire a cached one.

    Returns:
        ``(session, frame)``, where the frame is a ``KIND_HANDSHAKE_INIT`` for
        the responder. The session is **not** usable yet: pass the responder's
        confirmation to :meth:`Session.confirm` first.

    Raises:
        InvalidSignatureError: the bundle's signed prekey does not match the
            advertised identity key.
        RetiredSignedPrekey: the bundle is older than required.
    """
    engine = x3dh if x3dh is not None else X3DH()
    x3dh_session, init = engine.begin(
        identity_private, bundle, associated_data, minimum_signed_prekey_id
    )
    signed_prekey = bytes(bundle.signed_prekey)

    def build_ratchet() -> DoubleRatchet:
        return _initiator_ratchet(
            identity_private, signed_prekey, x3dh_session.root_key
        )

    session = Session(
        is_initiator=True,
        ratchet_factory=build_ratchet,
        x3dh_session=x3dh_session,
        initiator_identity=x3dh_session.initiator_identity,
        responder_identity=x3dh_session.responder_identity,
    )
    return session, encode_handshake_init(init)


def accept(
    responder: X3DHResponder,
    init_frame: bytes,
    *,
    associated_data: bytes = b'',
) -> Tuple['Session', bytes]:
    """
    Answer a handshake as the responder.

    The initiator's signature is verified and the referenced one-time prekey is
    spent before the shared secret is computed, so forged handshakes cannot
    drain the prekey pool.

    Args:
        responder: this side's responder, holding the identity key, the signed
            prekey and the prekey pool.
        init_frame: a ``KIND_HANDSHAKE_INIT`` frame from :func:`initiate`.
        associated_data: must equal what the initiator passed.

    Returns:
        ``(session, frame)``, where the frame is a ``KIND_KEY_CONFIRMATION`
        for the initiator. The session is usable immediately: the responder has
        nothing left to verify.

    Raises:
        WireError: the frame is malformed.
        InvalidSignatureError: the initiator did not prove ownership of the
            identity key it presented.
        OneTimePreKeyAlreadyUsed, NoSuchPreKey: the referenced prekey is spent
            or unknown. Forward secrecy is not negotiable here, so this
            propagates rather than falling back to the weaker 3-DH variant.
    """
    init = decode_handshake_init(init_frame)
    x3dh_session = responder.respond(init, associated_data)
    signed_prekey_private = responder.signed_prekey_private

    def build_ratchet() -> DoubleRatchet:
        return _responder_ratchet(signed_prekey_private, x3dh_session.root_key)

    session = Session(
        is_initiator=False,
        ratchet_factory=build_ratchet,
        x3dh_session=x3dh_session,
        initiator_identity=x3dh_session.initiator_identity,
        responder_identity=x3dh_session.responder_identity,
    )
    return session, encode_key_confirmation(
        x3dh_session.make_key_confirmation()
    )
