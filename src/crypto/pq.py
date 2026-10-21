"""
Post-quantum one-time prekeys: ML-KEM-768, and the discipline around them.

What this is for
----------------
X3DH's forward secrecy rests entirely on the one-time prekeys. A prekey that is
used twice is a prekey whose DH contribution is recoverable from two recorded
handshakes, so the whole "past messages stay unreadable" property collapses for
every session that shared it. That makes the one-time prekey pool the first thing
post-quantum has to strengthen: a harvest-now-decrypt-later adversary records the
X25519 handshake traffic today and decrypts it once X25519 is broken, and an
ML-KEM contribution that was part of the same root key is what makes that
harvest worthless.

So this is the same object as :class:`~src.crypto.prekey_store.OneTimePreKey` --
single use, reserved while advertised, spent exactly once -- with a different
primitive behind it. The discipline is the part that matters and it is
deliberately identical.

The primitive
-------------
ML-KEM-768 as implemented by ``cryptography``, which is why this module exists at
all: ``cryptography`` was a test-only dependency until PQXDH needed ML-KEM at
runtime, and moving it into ``requirements.in`` is a deliberate cost, not an
oversight. It is an audited, native implementation and writing KEM arithmetic
here instead would be a considerably worse trade than one more dependency.

Sizes, because they are protocol numbers rather than implementation details:
an encapsulation key is 1184 bytes, a shared secret is 32 bytes, a ciphertext is
1088 bytes, and a decapsulation key serialises to 64.

Two API details that cost time and are worth writing down
--------------------------------------------------------
**Private keys have no ``from_private_bytes``.** ``MLKEM768PrivateKey`` offers
``from_seed_bytes`` and ``private_bytes_raw`` and nothing else, so a private key
cannot be reloaded from the 64 raw bytes it serialises to by any other route.
The store therefore keeps those 64 bytes, and ``from_seed_bytes`` takes exactly
them -- not a 32-byte input, which is the obvious guess and is rejected with
"An ML-KEM-768 seed is 64 bytes long". Anyone reaching for a raw-bytes loader
will find it missing, and should know it is missing on purpose rather than by
oversight.

**Public keys must not be loaded through DER on the wire path.**
``serialization.load_der_public_key`` happily loads an Ed25519 or X25519 SPKI and
returns that type instead, so a bundle carrying a valid SPKI of the wrong
algorithm would produce an object that looks like a key and is not one.
``MLKEM768PublicKey.from_public_bytes`` takes the 1184 raw bytes, checks the
length itself and raises on anything else, so it is what the decoder uses, and
:func:`load_public_key` checks the type on top of that.
"""

import abc
import os
import threading
from typing import Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import mlkem

#: Encapsulation (public) key length, in bytes. Fixed by ML-KEM-768.
PUBLIC_KEY_LEN = 1184

#: Shared secret length, in bytes. Same as the X25519 DH output it is mixed with.
SHARED_SECRET_LEN = 32

#: Ciphertext length, in bytes. Travels in the initial handshake message.
CIPHERTEXT_LEN = 1088

#: Serialised form of a decapsulation (private) key, in bytes. This is also what
#: ``from_seed_bytes`` takes, which makes it the only representation a private
#: key can be reloaded from -- see the module docstring.
PRIVATE_KEY_LEN = 64


class PQError(Exception):
    """Base class for post-quantum prekey failures."""


class PQPreKeyExhausted(PQError):
    """
    Raised when a bundle is requested and the post-quantum pool is empty.

    A loud failure, and the same reasoning as
    :class:`~src.crypto.x3dh.PrekeyPoolExhausted`: a session that quietly drops
    its post-quantum contribution is a session whose forward secrecy will not
    survive a future break of ML-KEM, and the peer has no way to notice. Handing
    out such a bundle by default would make the downgrade invisible, so accepting
    it is an explicit decision by the caller.
    """


class PQPreKeyAlreadyUsed(PQError):
    """
    Raised when a post-quantum prekey that was already spent is referenced.

    Accepting it would destroy the forward secrecy the ML-KEM contribution
    exists to provide, which is the same trade the X25519 one-time prekey makes
    and for the same reason.
    """


class NoSuchPQPreKey(PQError, KeyError):
    """Raised when a referenced post-quantum prekey id is unknown."""


class InvalidPQKey(PQError):
    """Raised when key material is not usable ML-KEM-768."""


def load_public_key(raw: bytes) -> mlkem.MLKEM768PublicKey:
    """
    Load an encapsulation key from its 1184 wire bytes.

    The type check on top of the loader is not redundant paranoia: a DER
    SPKI of the wrong algorithm loads without complaint through
    ``serialization.load_der_public_key``, and the failure then surfaces as an
    attribute error somewhere far from the bundle that caused it.

    Raises:
        InvalidPQKey: the bytes are the wrong length or not a valid key.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise InvalidPQKey('an ML-KEM-768 public key must be bytes')
    raw = bytes(raw)
    if len(raw) != PUBLIC_KEY_LEN:
        raise InvalidPQKey(
            f'an ML-KEM-768 public key is {PUBLIC_KEY_LEN} bytes, got '
            f'{len(raw)}'
        )
    try:
        key = mlkem.MLKEM768PublicKey.from_public_bytes(raw)
    except Exception as exc:
        raise InvalidPQKey('not a usable ML-KEM-768 public key') from exc
    # Belt and braces: if a future cryptography adds a loader that accepts
    # something else, this is where it stops.
    if not isinstance(key, mlkem.MLKEM768PublicKey):
        raise InvalidPQKey('the loaded key is not an ML-KEM-768 public key')
    return key


def encapsulate(public_key: mlkem.MLKEM768PublicKey) -> Tuple[bytes, bytes]:
    """
    Produce a shared secret and its ciphertext against a peer's key.

    Returns:
        ``(shared_secret, ciphertext)``. Both are 32 and 1088 bytes, and the
        ciphertext has to reach the responder for it to decapsulate.

    Raises:
        InvalidPQKey: the key is not an ML-KEM-768 encapsulation key.
    """
    if not isinstance(public_key, mlkem.MLKEM768PublicKey):
        raise InvalidPQKey('encapsulate expects an MLKEM768PublicKey')
    shared, ciphertext = public_key.encapsulate()
    if len(shared) != SHARED_SECRET_LEN or len(ciphertext) != CIPHERTEXT_LEN:
        # Would mean the library changed the parameter set underneath us, and
        # every persisted prekey and every length constant here would be wrong.
        raise InvalidPQKey(
            f'unexpected ML-KEM sizes: {len(shared)} byte secret, '
            f'{len(ciphertext)} byte ciphertext'
        )
    return shared, ciphertext


def decapsulate(private_key: mlkem.MLKEM768PrivateKey,
                ciphertext: bytes) -> bytes:
    """
    Recover the shared secret from a peer's ciphertext.

    ML-KEM decapsulation does not fail on a wrong ciphertext: the
    ``decapsulate`` variant of the KEM returns a pseudorandom secret instead,
    which is what makes the scheme IND-CCA2. So a tampered ciphertext produces
    a *different* 32-byte secret rather than an exception, and the two sides
    derive different root keys. That is caught by the key confirmation and by
    the AEAD, not here, and pretending otherwise would be a claim about
    behaviour this library does not have.

    Raises:
        InvalidPQKey: the ciphertext is the wrong length, or the key is not an
            ML-KEM-768 decapsulation key.
    """
    if not isinstance(private_key, mlkem.MLKEM768PrivateKey):
        raise InvalidPQKey('decapsulate expects an MLKEM768PrivateKey')
    if not isinstance(ciphertext, (bytes, bytearray, memoryview)):
        raise InvalidPQKey('the ciphertext must be bytes')
    ciphertext = bytes(ciphertext)
    if len(ciphertext) != CIPHERTEXT_LEN:
        raise InvalidPQKey(
            f'an ML-KEM-768 ciphertext is {CIPHERTEXT_LEN} bytes, got '
            f'{len(ciphertext)}'
        )
    return private_key.decapsulate(ciphertext)


def generate_prekey(key_id: Optional[int] = None) -> 'PQOneTimePreKey':
    """
    Generate one post-quantum one-time prekey.

    Args:
        key_id: the id to publish it under, or None for a random 64-bit one.
    """
    private_key = mlkem.MLKEM768PrivateKey.generate()
    if key_id is None:
        key_id = int.from_bytes(os.urandom(8), 'big')
    return PQOneTimePreKey(key_id=key_id, private_key=private_key)


def stocked_pq_store(count: int = 4) -> 'InMemoryPQPreKeyStore':
    """
    A pool of ``count`` freshly generated prekeys, keyed 0..count-1.

    For a caller that needs a usable pool and has none yet -- a first run, or a
    test. A long-lived client wants :class:`PersistentPQPreKeyStore` instead,
    because a pool that lives in memory is a pool that is empty after a restart,
    and an empty pool means every later session is either refused or accepted
    without post-quantum protection.
    """
    if count < 1:
        raise PQError(f'a pool needs at least one prekey, asked for {count}')
    store = InMemoryPQPreKeyStore()
    for key_id in range(count):
        store.put(generate_prekey(key_id=key_id))
    return store


class PQOneTimePreKey:
    """
    One ML-KEM-768 decapsulation key and the id it is published under.

    Deliberately not a dataclass: the private key has no byte representation
    that can be reloaded (see the module docstring), so equality, repr and
    serialisation would all be misleading. :attr:`seed` is the form that
    persists.
    """

    __slots__ = ('key_id', '_private_key')

    def __init__(self, key_id: int, private_key: mlkem.MLKEM768PrivateKey):
        if isinstance(key_id, bool) or not isinstance(key_id, int):
            raise PQError('key_id must be an int')
        if key_id < 0 or key_id >= 2 ** 64:
            raise PQError('key_id must fit in 64 bits')
        if not isinstance(private_key, mlkem.MLKEM768PrivateKey):
            raise PQError('private_key must be an MLKEM768PrivateKey')
        self.key_id = key_id
        self._private_key = private_key

    @property
    def private_key(self) -> mlkem.MLKEM768PrivateKey:
        return self._private_key

    @property
    def public_key(self) -> mlkem.MLKEM768PublicKey:
        return self._private_key.public_key()

    @property
    def public_bytes(self) -> bytes:
        """The 1184 wire bytes, which is what a bundle carries."""
        return self.public_key.public_bytes_raw()

    @property
    def seed(self) -> bytes:
        """
        The 64 bytes this key is generated from and reloaded by.

        Named ``seed`` because that is what ``from_seed_bytes`` calls it, and it
        is a misnomer: 64 bytes is the expanded ``d || z`` form ML-KEM-768
        derives internally, not a 32-byte input. It is the whole private key:
        anyone holding it can decapsulate, so it belongs in the same sealed
        container as the X25519 private keys and nowhere else.
        """
        return self._private_key.private_bytes_raw()

    @classmethod
    def from_seed(cls, key_id: int, seed: bytes) -> 'PQOneTimePreKey':
        """
        Rebuild a prekey from a stored seed.

        Raises:
            PQError: the seed is the wrong length, or the key cannot be
                re-derived from it.
        """
        if not isinstance(seed, (bytes, bytearray, memoryview)):
            raise PQError('seed must be bytes')
        seed = bytes(seed)
        if len(seed) != PRIVATE_KEY_LEN:
            raise PQError(
                f'seed must be {PRIVATE_KEY_LEN} bytes, got {len(seed)}'
            )
        try:
            private_key = mlkem.MLKEM768PrivateKey.from_seed_bytes(seed)
        except Exception as exc:
            raise PQError('not a usable ML-KEM-768 seed') from exc
        return cls(key_id=key_id, private_key=private_key)

    def __repr__(self) -> str:
        # The id and the public half only. A repr that leaked key material would
        # end up in a log file, and a log file is somewhere keys go to die.
        return f'<PQOneTimePreKey id={self.key_id} public={self.public_bytes[:4].hex()}...>'

    def __eq__(self, other) -> bool:
        if not isinstance(other, PQOneTimePreKey):
            return NotImplemented
        return self.key_id == other.key_id and self.seed == other.seed

    def __hash__(self) -> int:
        return hash((self.key_id, self.seed))


class PQPreKeyStore(abc.ABC):
    """
    A pool of post-quantum one-time prekeys.

    The interface mirrors :class:`~src.crypto.prekey_store.PreKeyStore` method
    for method, including ``reserve_next`` and ``release``, because the failure
    this guards against is the same one in both pools: a bundle advertised twice
    hands out the same prekey twice, and the second handshake is refused -- a
    denial of service against a legitimate peer caused by nothing more than a
    naive server.
    """

    @abc.abstractmethod
    def put(self, prekey: PQOneTimePreKey) -> None:
        """Add a prekey to the pool."""

    @abc.abstractmethod
    def consume(self, key_id: int) -> PQOneTimePreKey:
        """
        Take a prekey out of circulation permanently.

        Raises:
            NoSuchPQPreKey: the id is unknown.
            PQPreKeyAlreadyUsed: the id was already spent.
        """

    @abc.abstractmethod
    def peek(self, key_id: int) -> PQOneTimePreKey:
        """
        Read a prekey without spending it.

        Raises:
            NoSuchPQPreKey: the id is unknown.
        """

    @abc.abstractmethod
    def peek_any(self) -> Optional[PQOneTimePreKey]:
        """Any prekey, spent or not, or None if the pool is empty."""

    @abc.abstractmethod
    def reserve_next(self) -> Optional[PQOneTimePreKey]:
        """
        Take the next prekey out of circulation but keep it consumable.

        Returns None if the pool is empty. The key stays spendable, so the
        handshake the bundle was advertised for still succeeds; what changes is
        that a *second* bundle cannot carry it.
        """

    @abc.abstractmethod
    def release(self, key_id: int) -> None:
        """
        Return a reserved prekey to circulation.

        Raises:
            PQPreKeyAlreadyUsed: the key was already consumed.
            NoSuchPQPreKey: the id is unknown.
        """

    @abc.abstractmethod
    def count(self) -> int:
        """How many prekeys are still available."""

    @abc.abstractmethod
    def available_ids(self) -> List[int]:
        """Ids of the prekeys still available."""

    @abc.abstractmethod
    def was_consumed(self, key_id: int) -> bool:
        """Whether a prekey was spent, and therefore whether it is gone."""


class InMemoryPQPreKeyStore(PQPreKeyStore):
    """
    A post-quantum prekey pool in memory.

    Single-process and not thread-safe against another *process*, exactly like
    :class:`~src.crypto.prekey_store.InMemoryPreKeyStore`. It does hold a lock,
    so concurrent threads cannot spend the same key twice: ``consume`` checks
    and deletes inside the lock, so two threads racing for the last prekey leave
    one winner and one :class:`PQPreKeyAlreadyUsed`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prekeys: Dict[int, PQOneTimePreKey] = {}
        self._consumed: set = set()
        self._reserved: set = set()

    def put(self, prekey: PQOneTimePreKey) -> None:
        with self._lock:
            if not isinstance(prekey, PQOneTimePreKey):
                raise PQError('put expects a PQOneTimePreKey')
            if prekey.key_id in self._prekeys or prekey.key_id in self._consumed:
                raise PQError(f'prekey {prekey.key_id} is already in the pool')
            self._prekeys[prekey.key_id] = prekey

    def consume(self, key_id: int) -> PQOneTimePreKey:
        with self._lock:
            if key_id in self._consumed:
                raise PQPreKeyAlreadyUsed(
                    f'post-quantum prekey {key_id} was already used; accepting '
                    f'it would destroy the forward secrecy the ML-KEM '
                    f'contribution exists to provide'
                )
            try:
                prekey = self._prekeys.pop(key_id)
            except KeyError:
                raise NoSuchPQPreKey(
                    f'unknown post-quantum prekey {key_id}'
                ) from None
            self._consumed.add(key_id)
            self._reserved.discard(key_id)
            return prekey

    def peek(self, key_id: int) -> PQOneTimePreKey:
        with self._lock:
            if key_id in self._consumed:
                raise NoSuchPQPreKey(
                    f'post-quantum prekey {key_id} was already used'
                )
            try:
                return self._prekeys[key_id]
            except KeyError:
                raise NoSuchPQPreKey(
                    f'unknown post-quantum prekey {key_id}'
                ) from None

    def peek_any(self) -> Optional[PQOneTimePreKey]:
        """
        An available prekey, without consuming or reserving it.

        Reserved keys are skipped, matching
        :meth:`~src.crypto.prekey_store.InMemoryPreKeyStore.peek_any`. A caller
        asking which key to hand out must be told about one that is not already
        advertised, or two bundles would carry the same prekey and the second
        handshake would be refused -- a denial of service against a legitimate
        peer, caused by nothing more than a naive server.
        """
        with self._lock:
            for key_id, prekey in self._prekeys.items():
                if key_id not in self._reserved:
                    return prekey
            return None

    def reserve_next(self) -> Optional[PQOneTimePreKey]:
        """
        The next available prekey, held out of circulation but still spendable.

        Reserved keys are skipped, so two consecutive calls hand out two
        different prekeys.
        """
        with self._lock:
            for key_id, prekey in self._prekeys.items():
                if key_id not in self._reserved:
                    self._reserved.add(key_id)
                    return prekey
            return None

    def release(self, key_id: int) -> None:
        with self._lock:
            if key_id in self._consumed:
                raise PQPreKeyAlreadyUsed(
                    f'post-quantum prekey {key_id} was already used'
                )
            if key_id not in self._prekeys:
                raise NoSuchPQPreKey(
                    f'unknown post-quantum prekey {key_id}'
                )
            self._reserved.discard(key_id)

    def reserved_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._reserved)

    def count(self) -> int:
        with self._lock:
            return len(self._prekeys)

    def available_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._prekeys)

    def was_consumed(self, key_id: int) -> bool:
        with self._lock:
            return key_id in self._consumed

    def replenish(self, low_water_mark: int, target: int = None) -> int:
        """
        Top the pool up to ``target``, and report how many were added.

        Raises:
            PQPreKeyExhausted: a non-positive target was requested.
        """
        target = target if target is not None else low_water_mark * 2
        if target <= 0:
            raise PQPreKeyExhausted('a replenishment target must be positive')
        with self._lock:
            missing = target - len(self._prekeys)
            added = 0
            while len(self._prekeys) < target:
                candidate = generate_prekey()
                if candidate.key_id in self._prekeys or \
                        candidate.key_id in self._consumed:
                    continue
                self._prekeys[candidate.key_id] = candidate
                added += 1
            return max(added, missing if added else 0)
