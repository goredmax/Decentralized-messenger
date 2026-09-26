"""
Persistent key and session material.

The in-memory stores in ``src/crypto`` are fine for tests and useless in
practice: the prekey pool dies with the process, so after a restart the only
reachable handshake is the 3-DH variant, which has no forward secrecy. This
module puts the same data behind :class:`~src.storage.container.Container`, so
it is sealed on disk and the prekey pool survives a restart.

What is and is not protected is worth being precise about:

- the file is encrypted and authenticated, so a reader without the password
  learns nothing, and a writer who is not us is detected
- single-use prekey enforcement is atomic within one process, guarded by a lock
  and a read-modify-write of the whole file
- it is **not** atomic across processes. Two processes opening the same
  container can each consume the same prekey. Enforcing that needs file locking,
  which is not implemented; the single-process guarantee is stated rather than
  implied
- an attacker who can replace the whole file with an older valid one is not
  detected, because there is no anchor outside the file
"""

import json
import os
import threading
from typing import Dict, List, Optional

from nacl.public import PrivateKey, PublicKey
from nacl.signing import SigningKey

from ..crypto.prekey_store import (
    NoSuchPreKey,
    OneTimePreKey,
    OneTimePreKeyAlreadyUsed,
    PreKeyStore,
    PreKeyStoreError,
)
from .container import Container, StorageError

#: Refuse to write a pool larger than this, so a corrupt file cannot make the
#: store try to allocate unbounded memory.
MAX_PREKEYS = 100_000


def _b64(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode('ascii')


def _unb64(text: str, what: str) -> bytes:
    import base64
    try:
        return base64.b64decode(text.encode('ascii'), validate=True)
    except Exception as exc:
        raise StorageError(f'{what} is not valid base64') from exc


class PersistentPreKeyStore(PreKeyStore):
    """
    A one-time prekey pool held in an encrypted container.

    The whole pool is rewritten on every mutation. That is O(n) per handshake,
    which is acceptable for the pool sizes in play and keeps the on-disk format
    trivially correct; a large deployment would want an append-only log with
    periodic compaction.
    """

    def __init__(self, container: Container, minimum_epoch: Optional[int] = None):
        self._container = container
        self._minimum_epoch = minimum_epoch
        self._lock = threading.Lock()
        self._keys: Dict[int, OneTimePreKey] = {}
        self._used: Dict[int, bool] = {}
        self._epoch = 0
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        epoch, raw = self._container.load(minimum_epoch=self._minimum_epoch)
        self._epoch = epoch
        try:
            document = json.loads(raw.decode('utf-8')) if raw else {}
        except (ValueError, UnicodeDecodeError) as exc:
            raise StorageError('prekey store payload is not valid JSON') from exc
        if not isinstance(document, dict):
            raise StorageError('prekey store payload must be an object')

        keys: Dict[int, OneTimePreKey] = {}
        for entry in document.get('keys') or []:
            key_id = int(entry['id'])
            private = PrivateKey(_unb64(entry['private'], 'prekey private key'))
            public = PublicKey(_unb64(entry['public'], 'prekey public key'))
            if bytes(private.public_key) != bytes(public):
                raise StorageError(
                    f'prekey {key_id} public key does not match its private key'
                )
            keys[key_id] = OneTimePreKey(
                key_id=key_id, private_key=private, public_key=public
            )
        used = {int(value) for value in document.get('used') or []}
        self._keys = keys
        self._used = {key_id: True for key_id in used}

    def _flush(self) -> None:
        document = {
            'keys': [
                {
                    'id': key.key_id,
                    'private': _b64(bytes(key.private_key)),
                    'public': _b64(bytes(key.public_key)),
                }
                for key in self._keys.values()
            ],
            'used': sorted(self._used),
        }
        self._epoch += 1
        self._container.store(
            json.dumps(document, sort_keys=True).encode('utf-8'), self._epoch
        )

    # ------------------------------------------------------------------
    # PreKeyStore
    # ------------------------------------------------------------------

    def put(self, key: OneTimePreKey) -> None:
        with self._lock:
            if key.key_id in self._used:
                raise OneTimePreKeyAlreadyUsed(
                    f'prekey id {key.key_id} was already consumed'
                )
            if key.key_id in self._keys:
                raise PreKeyStoreError(f'prekey id {key.key_id} already present')
            if len(self._keys) >= MAX_PREKEYS:
                raise PreKeyStoreError(f'prekey pool is full ({MAX_PREKEYS})')
            self._keys[key.key_id] = key
            try:
                self._flush()
            except Exception:
                # Do not leave an in-memory key the file does not have.
                del self._keys[key.key_id]
                raise

    def consume(self, key_id: int) -> OneTimePreKey:
        with self._lock:
            if key_id in self._used:
                raise OneTimePreKeyAlreadyUsed(
                    f'prekey id {key_id} was already consumed'
                )
            key = self._keys.get(key_id)
            if key is None:
                raise NoSuchPreKey(f'unknown prekey id {key_id}')

            del self._keys[key_id]
            self._used[key_id] = True
            try:
                self._flush()
            except Exception:
                # Put it back: a failed write must not consume a prekey, or a
                # transient disk error would silently destroy forward secrecy.
                self._used.pop(key_id, None)
                self._keys[key_id] = key
                raise
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

    def available_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._keys)

    def was_consumed(self, key_id: int) -> bool:
        with self._lock:
            return key_id in self._used

    def replenish(self, low_water_mark: int, target: Optional[int] = None) -> int:
        """
        Top the pool up, generating and persisting new keys as needed.

        Cheap to call after every handshake: it does nothing while the pool is
        at or above ``low_water_mark``.
        """
        from ..crypto.prekey_store import MAX_PREKEYS, make_one_time_prekey
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

    @property
    def epoch(self) -> int:
        """Current epoch, for a caller that wants to anchor it externally."""
        with self._lock:
            return self._epoch


class IdentityStore:
    """
    The long-term identity key and the active signed prekey.

    Held in their own container so that reading a session does not require
    decrypting the identity key, and so the two can be rotated independently.
    """

    def __init__(self, container: Container, minimum_epoch: Optional[int] = None):
        self._container = container
        self._minimum_epoch = minimum_epoch
        self._lock = threading.Lock()
        self._identity: Optional[SigningKey] = None
        self._signed_prekey: Optional[PrivateKey] = None
        self._signed_prekey_id: int = 0
        self._epoch = 0
        self._load()

    def _load(self) -> None:
        epoch, raw = self._container.load(minimum_epoch=self._minimum_epoch)
        self._epoch = epoch
        if not raw:
            return
        try:
            document = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise StorageError('identity payload is not valid JSON') from exc
        if 'identity' in document:
            self._identity = SigningKey(
                _unb64(document['identity'], 'identity key')
            )
        if 'signed_prekey' in document:
            self._signed_prekey = PrivateKey(
                _unb64(document['signed_prekey'], 'signed prekey')
            )
        self._signed_prekey_id = int(document.get('signed_prekey_id') or 0)

    def _flush(self) -> None:
        document = {}
        if self._identity is not None:
            document['identity'] = _b64(bytes(self._identity))
        if self._signed_prekey is not None:
            document['signed_prekey'] = _b64(bytes(self._signed_prekey))
            document['signed_prekey_id'] = self._signed_prekey_id
        self._epoch += 1
        self._container.store(
            json.dumps(document, sort_keys=True).encode('utf-8'), self._epoch
        )

    def initialise(self) -> None:
        """Generate a fresh identity key and signed prekey."""
        with self._lock:
            if self._identity is not None:
                raise StorageError('identity already initialised')
            self._identity = SigningKey.generate()
            self._signed_prekey = PrivateKey.generate()
            self._signed_prekey_id = 1
            self._flush()

    def rotate(self) -> int:
        """
        Replace the signed prekey, keeping the identity key.

        Peers that fetched the old bundle keep a valid signature over it, since
        the identity key did not change. Retiring it therefore needs the
        monotonic ``signed_prekey_id`` published in the new bundle, and a peer
        that requires a minimum id. This method returns the new id.

        Returns:
            The new ``signed_prekey_id``.
        """
        with self._lock:
            if self._identity is None:
                raise StorageError('identity not initialised')
            self._signed_prekey = PrivateKey.generate()
            self._signed_prekey_id += 1
            self._flush()
            return self._signed_prekey_id

    @property
    def signed_prekey_id(self) -> int:
        with self._lock:
            if self._signed_prekey is None:
                raise StorageError('identity not initialised')
            return self._signed_prekey_id

    @property
    def identity_key(self) -> SigningKey:
        with self._lock:
            if self._identity is None:
                raise StorageError('identity not initialised')
            return self._identity

    @property
    def signed_prekey_private(self) -> PrivateKey:
        with self._lock:
            if self._signed_prekey is None:
                raise StorageError('identity not initialised')
            return self._signed_prekey

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch


class SessionStore:
    """
    Double Ratchet states, keyed by peer.

    Each session is stored as its own container file, so one corrupt or
    unreadable conversation cannot take down the rest, and a session can be
    deleted without touching anything else.
    """

    def __init__(self, directory: str, password: str,
                 minimum_epoch: Optional[int] = None):
        self._directory = directory
        self._password = password
        self._minimum_epoch = minimum_epoch
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)

    def _path(self, peer: str) -> str:
        # The peer id is hashed, not used as a filename, so a hostile name
        # cannot escape the directory or pick a suffix.
        import hashlib
        digest = hashlib.sha256(peer.encode('utf-8')).hexdigest()[:32]
        return os.path.join(self._directory, f'{digest}.session')

    def save(self, peer: str, state) -> None:
        """Serialise a SessionState. Message keys are not written."""
        with self._lock:
            path = self._path(peer)
            epoch = 0
            if Container.exists(path):
                epoch = Container.open(path, self._password).load(
                    minimum_epoch=self._minimum_epoch
                )[0]
            import json
            document = json.dumps(state.serialize(), sort_keys=True).encode('utf-8')
            container = (
                Container.open(path, self._password)
                if Container.exists(path)
                else Container.create(path, self._password)
            )
            container.store(document, epoch + 1)

    def load(self, peer: str, expected_protocol_version: int):
        """
        Load a SessionState for ``peer``.

        Raises:
            KeyError: no session is stored for that peer.
        """
        from ..crypto.double_ratchet import IncompatibleSessionError
        path = self._path(peer)
        if not Container.exists(path):
            raise KeyError(f'no session for {peer!r}')
        _, raw = Container.open(path, self._password).load(
            minimum_epoch=self._minimum_epoch
        )
        import json
        try:
            document = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise StorageError('session payload is not valid JSON') from exc
        if document.get('version') != expected_protocol_version:
            raise IncompatibleSessionError(
                f'session version {document.get("version")} != '
                f'{expected_protocol_version}'
            )
        from ..crypto.double_ratchet import SessionState
        return SessionState.deserialize(document)

    def delete(self, peer: str) -> None:
        with self._lock:
            path = self._path(peer)
            if Container.exists(path):
                Container.open(path, self._password).delete()

    def peers(self) -> List[str]:
        """Session file names. Peer ids are hashed, so these are opaque."""
        with self._lock:
            return sorted(
                name for name in os.listdir(self._directory)
                if name.endswith('.session')
            )
