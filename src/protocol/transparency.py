"""
Key transparency for sender keys: a signed, hash-linked log of chain commitments.

What this is for
----------------
Sender Keys are trust on first use, and the module docstring of
:mod:`src.protocol.sender_key` says so plainly. The first distribution message
from a sender in a group is accepted and remembered; a later one with a different
chain id is refused and has to be confirmed by a human. That is better than
accepting whatever arrives, and it leaves two holes.

**A replay of the chain we already hold is invisible.** Trust on first use
recognises a chain id it has seen, so the server can roll a member back onto the
*same* chain it already trusts -- which is the chain that was compromised. The
first distribution looks fine, so nothing ever says otherwise.

**A rotation has to be confirmed by hand.** Every member who joins a group, or
who witnesses a removal, has to be told out of band that a chain changed. That
is a social process, and social processes fail at inconvenient times.

A log fixes both, for the same reason: it records every chain a sender has
published, in an order that cannot be rearranged, signed by the sender rather
than by whoever stores it. A replay is then visible because the log has moved on
past it, and a rotation is visible because the log says so.

The property, stated exactly
---------------------------
An entry commits to ``(index, group, sender, chain_id, iteration)`` and to the
hash of the entry before it. Each is signed by the *sender's* identity key, so
the log is verifiable by anyone and depends on nobody who stores it. A client
checks, for every entry it is shown:

- the signature is the sender's;
- the index is the one the log claims, so entries cannot be dropped silently --
  a gap shows up as a link that does not connect;
- the ``previous`` hash matches the entry the client already held at that
  position, so a fork is caught;
- the position is not *behind* the client's own anchor, so a log truncated to an
  earlier point in history is caught even though it is internally consistent.

What this does not give
-----------------------
**It is not key transparency, and the difference is not cosmetic.** Key
transparency means a client can *learn* what it has not been shown -- which
requires the log's hashes to reach it by some path other than the log's own
server: independent witnesses, several appenders, a gossip network, or an
epoch-based key directory. None of that exists here. What exists is:

- **Detected:** replay, including a replay of the currently trusted chain;
  a fork, where two members are shown different histories;
  a rollback to an earlier consistent point;
  a gap, where entries were removed from the middle;
  a forged entry, which will not verify.
- **Not detected:** a log that simply stops growing. Truncating the *tail* leaves
  a shorter, internally consistent log, and a client that has no anchor has
  nothing to compare against. An anchor is the mitigation and it is only as good
  as its own storage.
- **Not prevented:** a client with no anchor yet, on its first contact with a
  sender, can be shown any chain and must trust it. The log raises the cost of
  that only if the client keeps an anchor across sessions, which is the caller's
  job -- see :meth:`TransparencyLog.anchor` and
  :class:`~src.storage.store.PeerDirectory` for the other half of that.

So the honest one-line version: this detects and localises rollback and
forgery, and it removes the need for a human to confirm rotations, but it cannot
make a withholding server reveal itself. Calling it transparency without that
sentence would be the kind of claim this repository's other documents refuse to
make.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey

from ..crypto.x3dh import length_prefixed
from .sender_key import (
    AttestationStatus,
    ChainAttestation,
    InconsistentLog,
    SenderKeyDistributionMessage,
    SenderKeyId,
)

#: Domain separation from every other string hashed in this codebase.
TRANSPARENCY_LABEL = b"DecentralizedMessenger-Transparency-v1"

#: The ``previous`` hash of the first entry. All zero rather than empty, so an
#: entry can never claim to point at nothing by carrying an empty hash.
GENESIS = b'\x00' * 32

_HASH_LEN = 32
_SIG_LEN = 64


def _u32(value: int) -> bytes:
    return value.to_bytes(4, 'big')


class TransparencyError(Exception):
    """
    Base class for transparency log failures.

    Deliberately *not* a :class:`~src.protocol.sender_key.SenderKeyError`
    subclass: it cannot be one, because this module imports that one and a
    subclass relationship would be circular. The two are caught separately, and
    :class:`~src.protocol.sender_key.InconsistentLog` is the one that means "the
    log is unusable" to a caller holding a builder.
    """


class LogFormatError(TransparencyError):
    """A log entry is structurally wrong."""


class LogLinkError(InconsistentLog):
    """Two entries do not connect, or an index does not follow."""


class LogSignatureError(InconsistentLog):
    """An entry's signature does not verify against the sender it names."""


class LogRollback(TransparencyError):
    """The log is behind the position a client has already reached."""


def _signed_bytes(
    index: int,
    key_id: SenderKeyId,
    chain_id: bytes,
    iteration: int,
    previous: bytes,
) -> bytes:
    """
    The canonical bytes covered by the signature and by the hash.

    Length-prefixed, so no two different field sets can produce the same byte
    string. Without that, an entry could be re-cut into a different set of fields
    and keep its signature -- the same reason the X3DH transcript is built this
    way, and the same reason it lives in one shared helper.

    A function rather than a method, because signing needs these bytes for an
    entry that does not exist yet. Building a placeholder object and reading its
    method would mean the placeholder has to satisfy the same validation as a
    real entry, which is a constraint with no purpose.
    """
    return length_prefixed(
        TRANSPARENCY_LABEL,
        _u32(index),
        key_id.group_id.encode('utf-8'),
        bytes(key_id.sender),
        bytes(chain_id),
        _u32(iteration),
        bytes(previous),
    )


@dataclass(frozen=True)
class LogEntry:
    """
    One signed commitment to a sender's chain, in a fixed position.

    Frozen because an entry is a statement about a moment: a mutable one could be
    edited in place and its hash, which is what the next entry commits to, would
    no longer match what it was when that commitment was made.
    """

    index: int
    key_id: SenderKeyId
    chain_id: bytes
    iteration: int
    previous: bytes
    signature: bytes

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise LogFormatError('index must be an int')
        if self.index < 0:
            raise LogFormatError(f'index must not be negative: {self.index}')
        if not isinstance(self.key_id, SenderKeyId):
            raise LogFormatError('key_id must be a SenderKeyId')
        if not isinstance(self.chain_id, (bytes, bytearray)) or \
                len(self.chain_id) != _HASH_LEN:
            raise LogFormatError(f'chain_id must be {_HASH_LEN} bytes')
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise LogFormatError('iteration must be an int')
        if self.iteration < 0:
            raise LogFormatError('iteration must not be negative')
        if not isinstance(self.previous, (bytes, bytearray)) or \
                len(self.previous) != _HASH_LEN:
            raise LogFormatError(f'previous must be {_HASH_LEN} bytes')
        if not isinstance(self.signature, (bytes, bytearray)) or \
                len(self.signature) != _SIG_LEN:
            raise LogFormatError(f'signature must be {_SIG_LEN} bytes')

    def signed_bytes(self) -> bytes:
        """The bytes this entry's signature and hash cover."""
        return _signed_bytes(
            self.index, self.key_id, self.chain_id, self.iteration,
            self.previous,
        )

    def hash(self) -> bytes:
        """What the next entry commits to."""
        return hashlib.sha256(self.signed_bytes()).digest()

    def sign(self, identity_key: SigningKey) -> 'LogEntry':
        """
        Sign with the sender's own identity key.

        Signed by the sender rather than by whoever stores the log, and that is
        the whole design: the log is then verifiable independently of the party
        that could profit from serving a different one. A server signature would
        prove that the server said this, which is not worth knowing.

        Raises:
            LogFormatError: the signing key is not the one the entry names.
        """
        if bytes(identity_key.verify_key) != bytes(self.key_id.sender):
            raise LogFormatError(
                'the log entry names a different sender than the signing key'
            )
        signature = identity_key.sign(self.signed_bytes()).signature
        return LogEntry(
            index=self.index, key_id=self.key_id, chain_id=self.chain_id,
            iteration=self.iteration, previous=self.previous,
            signature=signature,
        )

    def verify(self) -> None:
        """
        Check the signature against the sender the entry names.

        Raises:
            LogSignatureError: it does not verify.
        """
        try:
            self.key_id.sender.verify(self.signed_bytes(), self.signature)
        except BadSignatureError as exc:
            raise LogSignatureError(
                f'log entry {self.index} is not signed by the sender it names'
            ) from exc


class TransparencyLog:
    """
    An append-only, hash-linked log in memory.

    The in-memory part is the whole implementation. There is no server here, no
    storage format and no replication, because there is no network to replicate
    over -- the same reason there is no real transport. What a client can check
    is checked here, so that when a server does exist the interesting part is
    already right.
    """

    def __init__(self) -> None:
        self._entries: List[LogEntry] = []
        self._by_index: Dict[int, LogEntry] = {}
        self._by_sender: Dict[SenderKeyId, int] = {}

    # ------------------------------------------------------------------
    # Appending
    # ------------------------------------------------------------------

    def append(
        self,
        identity_key: SigningKey,
        key_id: SenderKeyId,
        *,
        chain_id: bytes,
        iteration: int,
    ) -> LogEntry:
        """
        Record a new commitment for a sender's chain.

        The link is taken from the current head, so a caller cannot insert an
        entry anywhere else: appending out of order would be visible to every
        client that holds the neighbouring entries, but there is no reason to
        offer the option.

        Raises:
            LogSignatureError: the entry could not be built, which means the
                signing key is not the sender named in ``key_id``.
        """
        previous = self.head_hash()
        unsigned = LogEntry(
            index=len(self._entries), key_id=key_id, chain_id=chain_id,
            iteration=iteration, previous=previous,
            signature=b'\x00' * _SIG_LEN,
        )
        signed = unsigned.sign(identity_key)
        self._entries.append(signed)
        self._by_index[signed.index] = signed
        self._by_sender[key_id] = signed.index
        return signed

    def append_distribution(
        self, identity_key: SigningKey, message: SenderKeyDistributionMessage
    ) -> LogEntry:
        """
        Record the chain a distribution message carries.

        The message is verified first: an entry that committed to a chain nobody
        signed would let a storage party invent a rotation.
        """
        message.verify()
        return self.append(
            identity_key,
            message.key_id,
            chain_id=message.chain_id,
            iteration=message.iteration,
        )

    @classmethod
    def from_entries(cls, entries: Iterable[LogEntry]) -> 'TransparencyLog':
        """
        Build a log from entries a client was handed.

        Every entry is verified before it is stored, and the links are checked in
        order, so a log assembled from untrusted input either comes out whole and
        correct or raises. The alternative -- trusting the entries because they
        came in a list -- is the defect this module exists to avoid.

        Raises:
            LogSignatureError: an entry is not signed by the sender it names.
            LogLinkError: the entries do not connect, or an index does not follow.
        """
        log = cls()
        entries = list(entries)
        for position, entry in enumerate(entries):
            if entry.index != position:
                raise LogLinkError(
                    f'entry at position {position} claims index {entry.index}'
                )
            entry.verify()
        previous = GENESIS
        for entry in entries:
            if entry.previous != previous:
                raise LogLinkError(
                    f'entry {entry.index} does not link to the entry before it'
                )
            previous = entry.hash()
        log._entries = entries
        log._by_index = {entry.index: entry for entry in entries}
        for entry in entries:
            log._by_sender[entry.key_id] = entry.index
        return log

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._entries)

    def entries(self) -> Tuple[LogEntry, ...]:
        """Every entry, in order. A copy, so a caller cannot reorder the log."""
        return tuple(self._entries)

    def at(self, index: int) -> LogEntry:
        """
        The entry at a position.

        Raises:
            IndexError: the log is not that long, which is itself information a
                client needs rather than an error to swallow.
        """
        return self._entries[index]

    def head(self) -> Optional[LogEntry]:
        return self._entries[-1] if self._entries else None

    def head_hash(self) -> bytes:
        """What the next entry will commit to. ``GENESIS`` for an empty log."""
        head = self.head()
        return GENESIS if head is None else head.hash()

    def latest_for(self, key_id: SenderKeyId) -> Optional[LogEntry]:
        """The most recent entry for a sender, or None if it has none."""
        index = self._by_sender.get(key_id)
        return None if index is None else self._entries[index]

    def chains_for(self, key_id: SenderKeyId) -> List[bytes]:
        """Every chain id this log has ever seen from a sender, in order."""
        return [
            entry.chain_id for entry in self._entries
            if entry.key_id == key_id
        ]

    # ------------------------------------------------------------------
    # Verifying the log itself
    # ------------------------------------------------------------------

    def verify(self, *, anchor: Optional[Tuple[int, bytes]] = None) -> None:
        """
        Check the whole log, and optionally that it is not behind ``anchor``.

        Args:
            anchor: ``(index, entry_hash)`` the client reached before, from its
                own storage. A log that is shorter than that, or whose entry at
                that position hashes differently, has been rolled back.

        Raises:
            LogLinkError: two entries do not connect, or an index skips.
            LogSignatureError: an entry is not signed by the sender it names.
            LogRollback: the log is behind the anchor.
        """
        previous = GENESIS
        for position, entry in enumerate(self._entries):
            if entry.index != position:
                raise LogLinkError(
                    f'entry at position {position} claims index {entry.index}'
                )
            entry.verify()
            if entry.previous != previous:
                raise LogLinkError(
                    f'entry {entry.index} does not link to the entry before it'
                )
            previous = entry.hash()

        if anchor is not None:
            index, entry_hash = anchor
            if len(self._entries) <= index:
                raise LogRollback(
                    f'the log has {len(self._entries)} entries, but this client '
                    f'already reached position {index}'
                )
            if self._entries[index].hash() != entry_hash:
                raise LogRollback(
                    f'the log disagrees with this client at position {index}: '
                    f'it has been rewritten there'
                )

    def fork_points(self, other: 'TransparencyLog') -> List[int]:
        """
        Positions where two logs disagree, or ``[]`` if they agree.

        What a client would learn by comparing notes with another member, and the
        property a server cannot fake: two members shown different histories get
        the same list of divergent positions, and so does anyone who asks both.
        """
        points = []
        for index in range(min(self.size(), other.size())):
            if self.at(index).hash() != other.at(index).hash():
                points.append(index)
        return points


@dataclass(frozen=True)
class Anchor:
    """
    What a client remembers about how far it got.

    Small on purpose: an index and a hash. Anything more would be a second thing
    to roll back, and the value of an anchor is entirely in being independently
    held -- so the caller stores it somewhere the log's server cannot write, which
    is a requirement this module cannot meet for them.
    """

    index: int
    entry_hash: bytes

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) \
                or self.index < -1:
            raise LogFormatError('anchor index must be an int >= -1')
        if not isinstance(self.entry_hash, (bytes, bytearray)) or \
                len(self.entry_hash) != _HASH_LEN:
            raise LogFormatError(f'anchor hash must be {_HASH_LEN} bytes')

    @classmethod
    def genesis(cls) -> 'Anchor':
        """The anchor of a client that has never seen a log: nothing, at -1."""
        return cls(index=-1, entry_hash=GENESIS)

    def as_tuple(self) -> Tuple[int, bytes]:
        return self.index, self.entry_hash


class SenderKeyTransparency(ChainAttestation):
    """
    A client's view of the log, and the thing a session builder asks.

    Holds the entries this client has verified, per sender. A chain is current
    when it is the last one this client saw for that sender; stale when the log
    has moved on; unattested when the log has never mentioned it; and
    inconsistent when the entries do not hold together.

    The "last one this client saw" is the honest reading of what a one-sided log
    can support. It is not "the chain the log has", because a client is only ever
    shown part of a log, and claiming the latter would be claiming a guarantee
    the module docstring spends a paragraph saying does not exist.
    """

    def __init__(self, anchor: Optional[Anchor] = None):
        self._anchor = anchor or Anchor.genesis()
        self._entries: List[LogEntry] = []
        self._by_sender: Dict[SenderKeyId, int] = {}
        self._links: Dict[int, bytes] = {}
        self._changed = False

    # ------------------------------------------------------------------
    # Incorporating
    # ------------------------------------------------------------------

    def incorporate(self, entries: Iterable[LogEntry]) -> None:
        """
        Add entries, checking each one against what is already held.

        Checks, in order and before anything is stored: the signature is the
        sender's, the index follows, and the ``previous`` hash is either the
        genesis value or the hash of the entry that came before it. The first two
        catch a forged or misfiled entry; the third is what catches a fork and a
        gap.

        The links are checked against a working copy that includes the entries
        being added, so a batch that is internally consistent is accepted as a
        batch -- and one that is not is rejected without half of it landing.
        A half-incorporated log is worse than an unchanged one, because the
        caller would have no way to tell which entries are trusted.

        Raises:
            LogSignatureError: an entry is not signed by the sender it names.
            LogLinkError: an entry does not connect to what is held.
            LogRollback: an entry rewrites a position already incorporated.
        """
        pending = list(entries)
        seen = dict(self._links)

        for entry in pending:
            entry.verify()

            if entry.index in seen and seen[entry.index] != entry.hash():
                raise LogRollback(
                    f'position {entry.index} was already incorporated and this '
                    f'entry differs: the log has been rewritten there'
                )

            if entry.index == 0:
                expected = GENESIS
            elif entry.index - 1 in seen:
                expected = seen[entry.index - 1]
            elif entry.index - 1 >= self._anchor.index + 1:
                # The position before this one is past everything we have, so it
                # was skipped rather than merely unseen. An unseen entry before
                # the anchor is fine: the anchor already stands for it.
                raise LogLinkError(
                    f'entry {entry.index} skips position {entry.index - 1}, '
                    f'which is past the anchor this client holds'
                )
            else:
                expected = entry.previous

            if entry.index > 0 and entry.previous != expected:
                raise LogLinkError(
                    f'entry {entry.index} does not link to the entry before it'
                )
            if entry.index == 0 and entry.previous != GENESIS:
                raise LogLinkError('the first entry must point at nothing')

            seen[entry.index] = entry.hash()

        # Computed before anything is stored: an entry already held, and the same
        # entry, is the normal case rather than a change. Being re-shown what is
        # known happens on every poll, and appending it again would make the
        # view's size and its contents disagree with the log.
        new = [entry for entry in pending if entry.index not in self._links]
        for entry in new:
            self._links[entry.index] = entry.hash()
            self._entries.append(entry)
            self._by_sender[entry.key_id] = entry.index
        if new:
            self._anchor = Anchor(
                index=new[-1].index, entry_hash=new[-1].hash()
            )
            self._changed = True

    def adopt(self, log: TransparencyLog) -> None:
        """
        Take every entry from a log, after checking the log against the anchor.

        The anchor is checked against the *log*, before any of its entries are
        looked at. Checking only the entries it would add is not enough, and the
        difference is the whole attack: a log truncated to a point the client has
        already reached contains nothing new, so an entry-wise check accepts it
        and the client quietly carries on with a history that no longer contains
        the rotations it thought it had seen.

        Rolls this client forward only. A log that is behind the anchor is
        refused, because accepting it would forget what has already been seen.

        Raises:
            LogRollback: the log is behind the anchor, or differs at it.
            LogLinkError, LogSignatureError: the log does not hold together.
        """
        anchor = self._anchor.as_tuple() if self._anchor.index >= 0 else None
        log.verify(anchor=anchor)
        self.incorporate(log.entries())

    # ------------------------------------------------------------------
    # The attestation itself
    # ------------------------------------------------------------------

    def check(self, message: SenderKeyDistributionMessage) -> str:
        """
        Report whether ``message`` carries the chain the log has for its sender.

        Three states, and the middle one is the one that is easy to get wrong:

        - ``UNATTESTED`` when the log has never seen this chain id. It has no
          opinion, so the caller decides -- which is the behaviour from before the
          log existed, and is the right answer for a chain newer than anything
          the log holds.
        - ``CURRENT`` when this is the last chain the log has for the sender.
        - ``STALE`` when the log has seen this chain *and* a later one. Only
          then is "older" a statement the log is entitled to make: a chain it has
          never seen could be newer rather than older, and calling that stale
          would refuse a genuine rotation that simply has not been logged yet.

        Raises:
            SenderKeyError: the message does not verify, which is a property of
                the message rather than of the log.
        """
        message.verify()
        known = self.known_chains(message.key_id)
        if message.chain_id not in known:
            return AttestationStatus.UNATTESTED
        if known[-1] == message.chain_id:
            return AttestationStatus.CURRENT
        return AttestationStatus.STALE

    def _entry_at(self, index: int) -> LogEntry:
        for entry in reversed(self._entries):
            if entry.index == index:
                return entry
        raise LogLinkError(f'no entry at position {index}')

    # ------------------------------------------------------------------

    def anchor(self) -> Anchor:
        """
        The position this client has reached, to be stored independently.

        Useless unless it is stored somewhere the log's server cannot write, and
        that is not something this module can arrange. A client whose anchor
        lives next to the log has no rollback detection at all.
        """
        return self._anchor

    def has_moved(self) -> bool:
        """Whether anything has been incorporated since the last :meth:`anchor`."""
        return self._changed

    def known_chains(self, key_id: SenderKeyId) -> List[bytes]:
        """Chain ids this client has seen for a sender, in log order."""
        return [
            entry.chain_id for entry in self._entries
            if entry.key_id == key_id
        ]

    def size(self) -> int:
        return len(self._entries)

    def entries(self) -> Tuple[LogEntry, ...]:
        return tuple(self._entries)


def attested_builder():
    """
    A session builder with a log attached -- the shape a caller will want.

    Not used inside the library, and written out here so the intended wiring is
    a line rather than a paragraph: a member keeps a :class:`SenderKeyTransparency`
    next to its :class:`~src.protocol.sender_key.SenderKeySessionBuilder` and
    passes it to :meth:`~src.protocol.sender_key.SenderKeySessionBuilder.process`.
    """
    from .sender_key import SenderKeySessionBuilder

    return SenderKeySessionBuilder(), SenderKeyTransparency()


__all__ = [
    'Anchor',
    'ChainAttestation',
    'LogEntry',
    'LogFormatError',
    'LogLinkError',
    'LogRollback',
    'LogSignatureError',
    'SenderKeyTransparency',
    'TransparencyError',
    'TransparencyLog',
]
