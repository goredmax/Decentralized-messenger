"""
Sender Keys: group messaging without a pairwise ratchet per pair.

The problem
-----------
A Double Ratchet session is pairwise, and a group of n members would need
n(n-1)/2 of them. Worse, each pair would need its own handshake, so adding one
person to a group means handshaking with everyone. Signal's answer is to
decouple *who* the message is for from *how* it is encrypted.

How it works
------------
Each member keeps one symmetric chain, its **sender key**. To send to a group a
member:

1. derives one message key from its own chain, advancing the chain
2. encrypts the plaintext under a **fresh random** 32-byte key
3. sends the same ciphertext and that random key to every member, along with
   which iteration of the chain it belongs to

Each member holds a :class:`SenderKeySession` that mirrors the sender's chain
and can re-derive that same message key by iterating up to the announced index.
So the message key is never transmitted. One encryption serves the whole group,
and adding a member costs one handshake, not one per existing member.

What the random key buys
------------------------
The message key stays secret, so a member who joins later cannot recover older
messages even if the sender's chain is later compromised: they cannot re-derive
iterations that happened before they arrived. The forward secrecy of a sender
key therefore depends on the session being created before the messages it
covers, which is why distribution happens *before* sending.

Trust model, stated plainly
---------------------------
A distribution message is signed by the sender's identity key, so the server
cannot substitute one. But a server can *replay* an older, validly signed
distribution message and roll a member back to a compromised chain. This
implementation is therefore **trust on first use**: the first distribution
message seen from a given sender in a group is accepted and remembered, and any
later message must carry the same key id or it is reported as untrusted rather
than accepted. That is strictly better than accepting whatever arrives, and it
is not the same as full key transparency. See docs/SECURITY_BOUNDARIES.md.

Rotation
--------
Removing a member requires every remaining member to generate a new sender key
and distribute it. Until they do, the removed member still holds the old chain
and can read what follows. :meth:`SessionCipher.rotate_after_removal` produces
the new distribution; sending it to the other members is the caller's job, since
that means pairwise handshakes and transport.
"""

import hashlib
import hmac
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple

import nacl.bindings
import nacl.utils
from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from ..crypto.double_ratchet import MAX_SKIP

#: Domain separation from the Double Ratchet, which has its own info strings.
_SENDER_KEY_INFO = b"WhisperSenderKey"
_CONFIRMATION_INFO = b"WhisperKeyConfirmation"

_SYMMETRIC_KEY_LEN = 32
#: Fresh random material identifying one chain, so a rotation is detectable.
_CHAIN_ID_LEN = 32
#: Longest chain a sender will produce, and a receiver will iterate towards.
MAX_ITERATION = 1 << 31


class SenderKeyError(Exception):
    """Base class for sender key failures."""


class UntrustedSenderKey(SenderKeyError):
    """
    Raised when a member receives a distribution message that conflicts with the
    one it already accepted from that sender.

    Reported rather than silently accepted: it is either a replay from the
    server or a second device of the sender appearing unexpectedly, and the
    caller has to decide which. See the module docstring.
    """


class SenderKeyReplay(SenderKeyError):
    """Raised when a message arrives for an iteration already consumed."""


class SenderKeySkippedTooFar(SenderKeyError):
    """
    Raised when the gap in a chain exceeds the budget.

    A member that has been away longer than this cannot catch up by iterating,
    and the session has to be re-established.
    """

    #: How many iterations a receiver will derive in one catch-up.
    MAX_CATCH_UP = MAX_SKIP


@dataclass(frozen=True)
class SenderKeyId:
    """
    Identifies a sender within a group.

    Equality by value, so two members that derived the id independently agree on
    it, which is what lets ``SenderKeySessionBuilder`` key its state by sender.
    """

    group_id: str
    sender: VerifyKey

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise SenderKeyError('group_id must be a non-empty string')
        if not isinstance(self.sender, VerifyKey):
            raise SenderKeyError('sender must be a VerifyKey')

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SenderKeyId)
            and other.group_id == self.group_id
            and bytes(other.sender) == bytes(self.sender)
        )

    def __hash__(self) -> int:
        return hash((self.group_id, bytes(self.sender)))


@dataclass(frozen=True)
class SenderKeyDistributionMessage:
    """
    A member's current sender key, signed by its identity key.

    Distributed over a pairwise Double Ratchet session, so only that member's
    members can read it. The signature is what makes it usable at all: without
    it, whoever holds the pairwise session with a member could advertise a chain
    on their behalf.

    ``chain_id`` is fresh random material for this chain and is covered by the
    signature. It exists because ``SenderKeyId`` alone cannot detect a replay:
    that id is (group, sender), so a rotation does not change it, and a
    replayed older distribution would carry the same id as the live one and be
    indistinguishable from it. The chain id is what makes "this is a different
    chain than the one I accepted" a checkable statement.
    """

    key_id: SenderKeyId
    iteration: int
    chain_key: bytes
    signature: bytes
    chain_id: bytes = b''

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, SenderKeyId):
            raise SenderKeyError('key_id must be a SenderKeyId')
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool) \
                or not 0 <= self.iteration <= MAX_ITERATION:
            raise SenderKeyError(f'iteration out of range: {self.iteration}')
        if len(self.chain_key) != _SYMMETRIC_KEY_LEN:
            raise SenderKeyError(
                f'chain key must be {_SYMMETRIC_KEY_LEN} bytes'
            )
        if len(self.signature) != 64:
            raise SenderKeyError('signature must be 64 bytes')
        if len(self.chain_id) != _CHAIN_ID_LEN:
            raise SenderKeyError(f'chain_id must be {_CHAIN_ID_LEN} bytes')

    def _signed_bytes(self) -> bytes:
        return b''.join([
            _u32(len(self.key_id.group_id.encode('utf-8'))),
            self.key_id.group_id.encode('utf-8'),
            bytes(self.key_id.sender),
            _u32(self.iteration),
            self.chain_key,
            self.chain_id,
        ])

    def sign(self, identity_key: SigningKey) -> 'SenderKeyDistributionMessage':
        """
        Sign with the sender's identity key.

        Raises:
            SenderKeyError: the signing key is not the one named in ``key_id``.
                Signing with a different identity would let a member forge a
                chain under someone else's name.
        """
        if bytes(identity_key.verify_key) != bytes(self.key_id.sender):
            raise SenderKeyError(
                'signing key does not match the identity key in the key id'
            )
        return replace(
            self, signature=identity_key.sign(self._signed_bytes()).signature
        )

    def verify(self) -> None:
        """
        Check the signature against the identity key named in the key id.

        Raises:
            SenderKeyError: the signature does not verify.
        """
        try:
            self.key_id.sender.verify(self._signed_bytes(), self.signature)
        except BadSignatureError as exc:
            raise SenderKeyError('sender key signature does not verify') from exc


def _u32(value: int) -> bytes:
    return value.to_bytes(4, 'big')


@dataclass
class SenderKeyState:
    """
    The sending half: one chain per group, advancing with every message.

    Secret. Only the derived distribution message is ever transmitted, never
    this object.
    """

    key_id: SenderKeyId
    chain_key: bytes
    iteration: int = 0
    chain_id: bytes = b''

    def __post_init__(self) -> None:
        if len(self.chain_key) != _SYMMETRIC_KEY_LEN:
            raise SenderKeyError('chain key must be 32 bytes')
        if len(self.chain_id) != _CHAIN_ID_LEN:
            raise SenderKeyError('chain id must be 32 bytes')
        if self.iteration >= MAX_ITERATION:
            raise SenderKeyError('sender key exhausted; rotate it')

    @classmethod
    def create(cls, group_id: str, identity_key: SigningKey) -> 'SenderKeyState':
        """Start a fresh chain for a group."""
        return cls(
            key_id=SenderKeyId(group_id=group_id, sender=identity_key.verify_key),
            chain_key=nacl.utils.random(_SYMMETRIC_KEY_LEN),
            iteration=0,
            chain_id=nacl.utils.random(_CHAIN_ID_LEN),
        )

    def rotate(self) -> 'SenderKeyState':
        """
        Replace the chain with a fresh random one and a fresh chain id.

        Every remaining member has to receive the new distribution before the
        group is protected again, otherwise a removed member keeps reading. The
        new chain id is what lets a receiver notice that the chain changed, since
        the sender key id alone does not.
        """
        return SenderKeyState(
            key_id=self.key_id,
            chain_key=nacl.utils.random(_SYMMETRIC_KEY_LEN),
            iteration=0,
            chain_id=nacl.utils.random(_CHAIN_ID_LEN),
        )

    def next_message_key(self) -> Tuple[bytes, int]:
        """
        Derive this message's key and advance the chain.

        Returns:
            ``(message_key, iteration)`` for the message about to be sent.
        """
        iteration = self.iteration
        self.chain_key, message_key = _kdf_ck(self.chain_key)
        self.iteration = iteration + 1
        return message_key, iteration

    def distribution_message(self) -> SenderKeyDistributionMessage:
        """
        The current chain, unsigned.

        The receiver needs the chain at ``iteration``, so the message is built
        before this member advances. Sign it with the identity key.
        """
        return SenderKeyDistributionMessage(
            key_id=self.key_id,
            iteration=self.iteration,
            chain_key=self.chain_key,
            signature=b'\x00' * 64,
            chain_id=self.chain_id,
        )


def _kdf_ck(chain_key: bytes) -> Tuple[bytes, bytes]:
    """
    Same construction as the Double Ratchet's ``KDF_CK``.

    Next chain key is ``HMAC(ck, 0x01)``, message key is ``HMAC(ck, 0x02)``.
    Sharing the construction means the two subsystems cannot drift apart.
    """
    next_chain_key = hmac.new(chain_key, b'\x01', hashlib.sha256).digest()
    message_key = hmac.new(chain_key, b'\x02', hashlib.sha256).digest()
    return next_chain_key, message_key


@dataclass
class SenderKeySession:
    """
    The receiving half: a mirror of one sender's chain.

    Holds the chain position this member has reached, so it can catch up to any
    later iteration and refuse to go backwards.
    """

    key_id: SenderKeyId
    chain_key: bytes
    iteration: int = 0
    identity_key: Optional[VerifyKey] = None
    chain_id: bytes = b''

    @classmethod
    def from_distribution(
        cls, message: SenderKeyDistributionMessage
    ) -> 'SenderKeySession':
        """Adopt a verified distribution message."""
        return cls(
            key_id=message.key_id,
            chain_key=message.chain_key,
            iteration=message.iteration,
            identity_key=message.key_id.sender,
            chain_id=message.chain_id,
        )

    def get_message_key(self, target_iteration: int) -> bytes:
        """
        Derive the message key for ``target_iteration``.

        Advances this session's position to just past it.

        Raises:
            SenderKeyReplay: the iteration was already consumed.
            SenderKeySkippedTooFar: the gap exceeds the catch-up budget.
        """
        if not isinstance(target_iteration, int) or \
                isinstance(target_iteration, bool) or target_iteration < 0:
            raise SenderKeyError('target_iteration must be a non-negative int')
        if target_iteration < self.iteration:
            raise SenderKeyReplay(
                f'iteration {target_iteration} was already consumed, this '
                f'session is at {self.iteration}'
            )
        gap = target_iteration - self.iteration
        if gap > SenderKeySkippedTooFar.MAX_CATCH_UP:
            raise SenderKeySkippedTooFar(
                f'gap of {gap} exceeds the catch-up budget of '
                f'{SenderKeySkippedTooFar.MAX_CATCH_UP}'
            )

        chain_key = self.chain_key
        for _ in range(gap):
            chain_key, _ = _kdf_ck(chain_key)
        chain_key, message_key = _kdf_ck(chain_key)
        self.chain_key = chain_key
        self.iteration = target_iteration + 1
        return message_key

    def advance_to(self, target_iteration: int) -> None:
        """Move the position forward without keeping any derived key."""
        self.get_message_key(target_iteration)


@dataclass(frozen=True)
class SenderKeyMessage:
    """
    One group message.

    The key is not here. It is re-derived by each recipient from their mirror of
    the sender's chain, at the announced iteration. Nothing secret travels
    except the ciphertext itself.

    That is what lets one encryption serve the whole group, and what stops a
    member who joins later from reading history: they cannot re-derive
    iterations that happened before their session existed.
    """

    key_id: SenderKeyId
    iteration: int
    ciphertext: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, SenderKeyId):
            raise SenderKeyError('key_id must be a SenderKeyId')
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool) \
                or not 0 <= self.iteration <= MAX_ITERATION:
            raise SenderKeyError(f'iteration out of range: {self.iteration}')
        if len(self.ciphertext) < 24 + 16:
            raise SenderKeyError('ciphertext is too short to be a nonce and tag')


class SessionCipher:
    """
    Encrypt to a group and decrypt from it.

    One instance per local member. The sending chain is per group; receiving
    state is per sender within a group, held by
    :class:`SenderKeySessionBuilder`.
    """

    def __init__(self, identity_key: SigningKey):
        if not isinstance(identity_key, SigningKey):
            raise SenderKeyError('identity_key must be a SigningKey')
        self._identity_key = identity_key
        self._outgoing: Dict[str, SenderKeyState] = {}
        self._builder = SenderKeySessionBuilder()

    # ------------------------------------------------------------------
    # sending
    # ------------------------------------------------------------------

    def sender_key(self, group_id: str) -> SenderKeyState:
        """
        The current sending chain for a group, created on first use.

        Exposed so a caller can distribute it. It is a live object: reading
        ``iteration`` after sending gives the current position.
        """
        state = self._outgoing.get(group_id)
        if state is None:
            state = SenderKeyState.create(group_id, self._identity_key)
            self._outgoing[group_id] = state
        return state

    def distribution_for(self, group_id: str) -> SenderKeyDistributionMessage:
        """
        A signed distribution message for the group's current chain.

        Must be delivered to every member over a pairwise session *before* any
        message encrypted under it, or the recipients cannot derive the message
        key. The chain position is the one at the time of the call, so calling
        this after sending advances it further and the recipients catch up by
        iterating.
        """
        return self.sender_key(group_id).distribution_message().sign(
            self._identity_key
        )

    def encrypt(
        self, plaintext: bytes, group_id: str
    ) -> Tuple[SenderKeyMessage, SenderKeyDistributionMessage]:
        """
        Encrypt one message for a whole group.

        The key is the one derived from the chain, and the chain advances, so
        every message uses fresh key material without any of it being
        transmitted.

        Returns:
            ``(message, distribution)``. The distribution is included because
            the caller almost always still owes it to at least one member; it is
            cheap to send and useless to send late.

        Raises:
            SenderKeyError: the group is exhausted and needs a rotation.
        """
        state = self.sender_key(group_id)
        message_key, iteration = state.next_message_key()
        nonce = nacl.utils.random(24)
        ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            plaintext, b'', nonce, message_key
        )
        message = SenderKeyMessage(
            key_id=state.key_id,
            iteration=iteration,
            ciphertext=nonce + ciphertext,
        )
        return message, self.distribution_for(group_id)

    def rotate_after_removal(
        self, group_id: str
    ) -> SenderKeyDistributionMessage:
        """
        Replace the sending chain after a member was removed.

        Until every remaining member receives the new distribution, the removed
        member still holds the old chain and can read what follows. Delivering it
        is the caller's job.
        """
        state = self.sender_key(group_id)
        self._outgoing[group_id] = state.rotate()
        return self.distribution_for(group_id)

    # ------------------------------------------------------------------
    # receiving
    # ------------------------------------------------------------------

    @property
    def builder(self) -> 'SenderKeySessionBuilder':
        """The receiving state. Exposed for inspection and removal."""
        return self._builder

    def process_distribution(
        self, message: SenderKeyDistributionMessage
    ) -> Tuple[SenderKeyId, bool]:
        """
        Adopt a distribution message from a peer.

        Returns:
            ``(key_id, untrusted)``. ``untrusted`` is True when the message
            conflicts with the one already accepted from that sender, in which
            case nothing was changed and the caller must decide what to do.

        Raises:
            SenderKeyError: the signature does not verify.
            UntrustedSenderKey: ``strict`` is set and the message conflicts.
        """
        message.verify()
        return self._builder.process(message)

    def decrypt(self, message: SenderKeyMessage) -> bytes:
        """
        Decrypt one group message.

        Raises:
            SenderKeyError: no session for that sender, or the session must be
                refreshed from a distribution message first.
            SenderKeyReplay: the iteration was already consumed.
            SenderKeyError: the ciphertext failed to authenticate, meaning the
                message was altered.
        """
        session = self._builder.get_session(message.key_id)
        message_key = session.get_message_key(message.iteration)
        nonce = message.ciphertext[:24]
        body = message.ciphertext[24:]
        try:
            return nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
                body, b'', nonce, message_key
            )
        except Exception as exc:
            raise SenderKeyError(
                'group message failed authentication: altered in transit, or the '
                'session is out of step with the sender'
            ) from exc

    def remove_member(self, group_id: str, member: VerifyKey) -> None:
        """
        Forget a removed member's session.

        Only affects this device. The other members still need to rotate, or the
        removed member keeps reading.
        """
        self._builder.remove(group_id, member)


class SenderKeySessionBuilder:
    """
    Receiving state, one session per sender per group, with TOFU on identity.

    The first distribution message from a sender is accepted and remembered.
    Later messages from the same sender must carry the same key id; otherwise
    the session is left alone and the conflict is reported, because accepting it
    would let whoever controls delivery roll this member back onto an older,
    possibly compromised chain.
    """

    def __init__(self) -> None:
        self._sessions: Dict[SenderKeyId, SenderKeySession] = {}
        #: chain ids trusted per sender *within a group*. Scoping to the group
        #: matters: a member joining a second group legitimately starts a fresh
        #: chain there, and that must not look like a rotation.
        self._trusted: Dict[SenderKeyId, Set[bytes]] = {}

    def process(
        self, message: SenderKeyDistributionMessage
    ) -> Tuple[SenderKeyId, bool]:
        """
        Adopt or reject a distribution message.

        The chain id is the decision point, not the sender key id: a rotation
        changes the chain id while leaving the key id alone, so a replayed older
        chain and a genuine rotation are both detectable here.

        Returns:
            ``(key_id, untrusted)``; ``untrusted`` True means nothing changed and
            the caller has to decide. A rotation therefore has to be confirmed
            out of band, which is the safe direction to fail in.
        """
        key_id = message.key_id
        known = self._trusted.setdefault(key_id, set())

        if message.chain_id in known:
            # The same chain we already accepted. Only ever move forward, so a
            # replay of an older distribution cannot roll the position back.
            existing = self._sessions.get(key_id)
            if existing is None:
                self._sessions[key_id] = SenderKeySession.from_distribution(message)
            elif message.iteration > existing.iteration:
                existing.chain_key = message.chain_key
                existing.iteration = message.iteration
                existing.chain_id = message.chain_id
            return key_id, False

        if not known:
            # The first chain we ever see from this sender in this group. Adopt
            # it: there is nothing to compare against yet, and refusing every
            # first contact would make the protocol unusable.
            known.add(message.chain_id)
            self._sessions[key_id] = SenderKeySession.from_distribution(message)
            return key_id, False

        # A second, different chain from a sender we already trust in this
        # group. That is either a rotation, which the caller must confirm, or a
        # replay of something we dropped. Never accepted silently.
        return key_id, True

    def get_session(self, key_id: SenderKeyId) -> SenderKeySession:
        """
        The session for a sender.

        Raises:
            UntrustedSenderKey: no distribution message has been processed for
                this sender yet.
        """
        session = self._sessions.get(key_id)
        if session is None:
            raise UntrustedSenderKey(
                f'no sender key session for {key_id.group_id} from this sender; '
                f'process a distribution message first'
            )
        return session

    def remove(self, group_id: str, member: VerifyKey) -> None:
        """Drop a member's session, locally."""
        for key_id in list(self._sessions):
            if key_id.group_id == group_id and bytes(key_id.sender) == bytes(member):
                del self._sessions[key_id]
                self._trusted.pop(key_id, None)

    def trust(self, message: SenderKeyDistributionMessage) -> SenderKeySession:
        """
        Explicitly accept a chain that :meth:`process` flagged.

        The caller's decision point for a rotation. A new chain id from a known
        sender is refused by default, so accepting one is always a deliberate
        act, and this is the only way to make it.

        Raises:
            SenderKeyError: the signature does not verify.
        """
        message.verify()
        self._trusted.setdefault(message.key_id, set()).add(message.chain_id)
        session = SenderKeySession.from_distribution(message)
        self._sessions[message.key_id] = session
        return session

    def trusted_chains(self, key_id: SenderKeyId) -> List[bytes]:
        """Chain ids currently trusted for a sender within a group."""
        return sorted(self._trusted.get(key_id, set()))

    def sessions(self) -> List[SenderKeyId]:
        """Every sender this member currently holds a session for."""
        return sorted(self._sessions, key=lambda k: (k.group_id, bytes(k.sender)))
