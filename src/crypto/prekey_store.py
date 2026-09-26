"""
One-time prekey storage for X3DH.

X3DH is only forward secret while each one-time prekey (OPK) is used at most
once. A prekey that is handed out twice gives an attacker who captured both
sessions enough material to recover the shared secret of the earlier session,
so single use has to be an enforced storage guarantee, not a convention.

This module defines the storage contract plus a reference in-memory
implementation. A persistent implementation must make ``consume`` atomic with
respect to concurrent handshakes (single DELETE ... RETURNING, a transaction,
or an equivalent compare-and-delete) so two simultaneous handshakes can never
observe the same prekey as available.
"""

import abc
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from nacl.public import PrivateKey, PublicKey


class PreKeyStoreError(Exception):
    """Base class for prekey storage failures."""


class NoSuchPreKey(PreKeyStoreError, KeyError):
    """Raised when a prekey id is unknown to the store."""


class OneTimePreKeyAlreadyUsed(PreKeyStoreError):
    """
    Raised when a one-time prekey is requested that was already consumed.

    This is a security event, not a transient error: it means the prekey was
    either replayed by a peer or the store was consulted twice for one session.
    The handshake must be aborted.
    """


#: Ceiling on pool size, so a corrupt or hostile file cannot make a store
#: allocate without bound.
MAX_PREKEYS = 100_000


@dataclass(frozen=True)
class OneTimePreKey:
    """A single one-time prekey pair."""

    key_id: int
    private_key: PrivateKey
    public_key: PublicKey

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, int) or self.key_id < 0:
            raise ValueError('prekey id must be a non-negative int')
        if len(bytes(self.public_key)) != 32:
            raise ValueError('one-time prekey public key must be 32 bytes')


class PreKeyStore(abc.ABC):
    """
    Storage contract for one-time prekeys.

    Implementations must guarantee that :meth:`consume` is atomic: once it
    returns a key, no later call may return the same key again, even under
    concurrent access.
    """

    @abc.abstractmethod
    def put(self, key: OneTimePreKey) -> None:
        """Store a freshly generated one-time prekey."""

    @abc.abstractmethod
    def consume(self, key_id: int) -> OneTimePreKey:
        """
        Atomically fetch and delete the prekey with ``key_id``.

        Raises:
            NoSuchPreKey: the id is unknown (never existed, or already used).
            OneTimePreKeyAlreadyUsed: the id existed but was already consumed.
        """

    @abc.abstractmethod
    def peek(self, key_id: int) -> OneTimePreKey:
        """
        Read a prekey *without* consuming it.

        Used when publishing a bundle. Publication must not consume, because the
        handshake may never arrive, but the read must not resurrect a spent key.

        Raises:
            OneTimePreKeyAlreadyUsed: the id was already consumed.
            NoSuchPreKey: the id is unknown.
        """

    @abc.abstractmethod
    def peek_any(self) -> Optional[OneTimePreKey]:
        """Return some available prekey without consuming it, or ``None``."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return the number of prekeys still available for use."""

    def bulk_put(self, keys) -> None:
        """Convenience helper to seed a batch of prekeys."""
        for key in keys:
            self.put(key)

    def replenish(self, low_water_mark: int, target: int = None) -> int:
        """
        Top the pool up to ``target`` keys, generating new ones as needed.

        Does nothing while the pool is at or above ``low_water_mark``, so it is
        cheap to call after every handshake.

        Returns:
            How many keys were added.

        Raises:
            ValueError: the arguments are inconsistent, or ``target`` exceeds
                the implementation limit.
        """
        if low_water_mark < 0:
            raise ValueError('low_water_mark must be non-negative')
        target = low_water_mark if target is None else target
        if target < low_water_mark:
            raise ValueError('target must be at least low_water_mark')
        if target > MAX_PREKEYS:
            raise ValueError(f'target exceeds the {MAX_PREKEYS} limit')
        added = 0
        while self.count() < target:
            self.put(make_one_time_prekey())
            added += 1
        return added


class InMemoryPreKeyStore(PreKeyStore):
    """
    Thread-safe in-memory reference implementation.

    Intended for tests and for a single process that does not need durability.
    A persistent store should provide the same semantics on top of a database.
    """

    def __init__(self) -> None:
        self._used: Dict[int, bool] = {}
        self._keys: Dict[int, OneTimePreKey] = {}
        self._lock = threading.Lock()

    def put(self, key: OneTimePreKey) -> None:
        with self._lock:
            if key.key_id in self._used:
                raise OneTimePreKeyAlreadyUsed(
                    f'prekey id {key.key_id} was already consumed'
                )
            self._keys[key.key_id] = key

    def consume(self, key_id: int) -> OneTimePreKey:
        with self._lock:
            if key_id in self._used:
                raise OneTimePreKeyAlreadyUsed(
                    f'prekey id {key_id} was already consumed'
                )
            key = self._keys.pop(key_id, None)
            if key is None:
                raise NoSuchPreKey(f'unknown prekey id {key_id}')
            self._used[key_id] = True
            return key

    def peek(self, key_id: int) -> OneTimePreKey:
        with self._lock:
            if key_id in self._used:
                raise OneTimePreKeyAlreadyUsed(
                    f'prekey id {key_id} was already consumed'
                )
            key = self._keys.get(key_id)
            if key is None:
                raise NoSuchPreKey(f'unknown prekey id {key_id}')
            return key

    def peek_any(self) -> Optional[OneTimePreKey]:
        with self._lock:
            if not self._keys:
                return None
            return self._keys[min(self._keys)]

    def count(self) -> int:
        with self._lock:
            return len(self._keys)

    def available_ids(self):
        """Test helper: ids that are still consumable."""
        with self._lock:
            return sorted(self._keys)

    def was_consumed(self, key_id: int) -> bool:
        """Test helper: whether ``key_id`` has been used."""
        with self._lock:
            return key_id in self._used


def generate_prekey_id() -> int:
    """
    Return a fresh 64-bit prekey id.

    The Signal specification uses a monotonic counter per bundle. A counter
    keeps ids unique and ordered; a random 64-bit value is used here only
    because this module does not own persistence. A persistent store should
    replace this with a real counter to keep ids unique across restarts.
    """
    import os
    return int.from_bytes(os.urandom(8), 'big')


def make_one_time_prekey(key_id: Optional[int] = None) -> OneTimePreKey:
    """Generate a one-time prekey, with an id if none is supplied."""
    private_key = PrivateKey.generate()
    return OneTimePreKey(
        key_id=generate_prekey_id() if key_id is None else key_id,
        private_key=private_key,
        public_key=private_key.public_key,
    )
