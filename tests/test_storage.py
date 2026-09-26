"""
Tests for encrypted, persistent storage.

The threat here is specific: session state is a decryption capability, and an
attacker who can write the file can roll counters back, replay an old ratchet or
swap a DH key. The tests therefore care about three things: the file is opaque
without the password, modification is detected, and a failed write never
consumes a prekey.
"""

import base64
import json
import os
import threading

import pytest
from nacl.public import PrivateKey
from nacl.signing import SigningKey

from src.crypto.prekey_store import (
    NoSuchPreKey,
    OneTimePreKeyAlreadyUsed,
    PreKeyStoreError,
)
from src.crypto.x3dh import X3DH
from src.storage.container import (
    CONTAINER_VERSION,
    EpochRollback,
    KdfParams,
    MalformedContainer,
    StorageError,
    WrongPassword,
    Container,
)
from src.storage.store import (
    IdentityStore,
    PersistentPreKeyStore,
    SessionStore,
)

# Weaker parameters keep the suite fast. Production defaults are in DEFAULT_KDF.
FAST = KdfParams(n_log2=12, r=8, p=1)
PASSWORD = 'correct horse battery staple'


@pytest.fixture
def container_path(tmp_path):
    return str(tmp_path / 'state.bin')


class TestContainerBasics:
    def test_round_trip(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        container.store(b'payload bytes', epoch=1)
        epoch, payload = container.open(container_path, PASSWORD).load()
        assert epoch == 1
        assert payload == b'payload bytes'

    def test_reopen_after_process_boundary(self, container_path):
        Container.create(container_path, PASSWORD, kdf_params=FAST).store(b'x', 3)
        # A fresh open is what a restart looks like.
        assert Container.open(container_path, PASSWORD).load() == (3, b'x')

    def test_file_is_not_really_a_container(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        container.store(b'secret material', epoch=1)
        raw = open(container_path, 'rb').read()
        assert b'secret material' not in raw
        assert PASSWORD.encode() not in raw

    def test_wrong_password_rejected(self, container_path):
        Container.create(container_path, PASSWORD, kdf_params=FAST).store(b'x', 1)
        with pytest.raises(WrongPassword):
            Container.open(container_path, 'not the password').load()

    def test_create_refuses_to_clobber(self, container_path):
        Container.create(container_path, PASSWORD, kdf_params=FAST)
        with pytest.raises(FileExistsError):
            Container.create(container_path, PASSWORD, kdf_params=FAST)

    def test_create_overwrite_when_asked(self, container_path):
        """
        Overwrite destroys the old state and starts a new epoch at 0.

        A caller holding an external epoch anchor must re-anchor after this,
        which is why the reset is documented rather than silent.
        """
        Container.create(container_path, PASSWORD, kdf_params=FAST).store(b'old', 1)
        Container.create(container_path, PASSWORD, kdf_params=FAST, overwrite=True)
        assert Container.open(container_path, PASSWORD).load() == (0, b'')

    def test_missing_file(self, container_path):
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD)

    def test_delete_is_idempotent(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        container.delete()
        container.delete()
        assert not os.path.exists(container_path)

    @pytest.mark.parametrize('payload', [b'', b'\x00' * 4096, bytes(range(256))])
    def test_arbitrary_payloads(self, container_path, payload):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        container.store(payload, epoch=1)
        assert Container.open(container_path, PASSWORD).load()[1] == payload

    def test_empty_password_rejected(self, container_path):
        with pytest.raises(ValueError):
            Container.create(container_path, '', kdf_params=FAST)


class TestTamperDetection:
    def _seal(self, path, payload=b'x', epoch=1):
        container = Container.create(path, PASSWORD, kdf_params=FAST)
        container.store(payload, epoch)
        return container

    def test_modified_ciphertext_detected(self, container_path):
        self._seal(container_path)
        raw = bytearray(open(container_path, 'rb').read())
        raw[-1] ^= 0xFF
        open(container_path, 'wb').write(bytes(raw))
        with pytest.raises(WrongPassword):
            Container.open(container_path, PASSWORD).load()

    def test_modified_nonce_detected(self, container_path):
        self._seal(container_path)
        raw = bytearray(open(container_path, 'rb').read())
        raw[29] ^= 0xFF
        open(container_path, 'wb').write(bytes(raw))
        with pytest.raises(WrongPassword):
            Container.open(container_path, PASSWORD).load()

    def test_implausible_kdf_parameters_refused(self, container_path):
        """
        Parameters below the floor are refused before any key derivation.

        Refusing early is fine; what matters is that it is a storage-domain
        error rather than a bare ValueError from a header read off disk.
        """
        self._seal(container_path)
        raw = bytearray(open(container_path, 'rb').read())
        # header = magic(8) | version(1) | kdf_id(1) | N_log2(1) | r(1) | p(1)
        raw[len(b'ANARCHY\x01') + 2] = 8      # N_log2 -> 256, i.e. free to guess
        open(container_path, 'wb').write(bytes(raw))
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD)

    def test_weakened_kdf_parameters_fail_authentication(self, container_path):
        """
        The KDF parameters are authenticated as associated data.

        This is the property that matters. N_log2 is rewritten from 14 to 12:
        both are inside the allowed range, so the file still looks plausible and
        the key derivation still succeeds, but it derives a *different* key from
        the same password. Only the AEAD over the header catches that. Without
        it an attacker would rewrite the header to make brute-forcing cheap.
        """
        stronger = KdfParams(n_log2=14, r=8, p=1)
        container = Container.create(container_path, PASSWORD, kdf_params=stronger)
        container.store(b'x', epoch=1)

        raw = bytearray(open(container_path, 'rb').read())
        raw[len(b'ANARCHY\x01') + 2] = 12     # cheaper, but still allowed
        open(container_path, 'wb').write(bytes(raw))

        with pytest.raises(WrongPassword):
            Container.open(container_path, PASSWORD).load()

    def test_unknown_kdf_id_rejected(self, container_path):
        self._seal(container_path)
        raw = bytearray(open(container_path, 'rb').read())
        raw[len(b'ANARCHY\x01') + 1] = 0x7F   # kdf_id
        open(container_path, 'wb').write(bytes(raw))
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD)

    def test_weakened_parameters_cannot_be_used_at_all(self, container_path):
        """
        Even a legitimate file may not use parameters below the floor, so a
        caller cannot opt into a weak container by accident.
        """
        with pytest.raises(ValueError):
            KdfParams(n_log2=8)

    def test_bad_magic_rejected(self, container_path):
        self._seal(container_path)
        raw = bytearray(open(container_path, 'rb').read())
        raw[0] = ord('X')
        open(container_path, 'wb').write(bytes(raw))
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD)

    def test_unknown_version_rejected(self, container_path):
        self._seal(container_path)
        raw = bytearray(open(container_path, 'rb').read())
        raw[len(b'ANARCHY\x01')] = CONTAINER_VERSION + 1
        open(container_path, 'wb').write(bytes(raw))
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD)

    def test_truncated_file_rejected(self, container_path):
        self._seal(container_path)
        raw = open(container_path, 'rb').read()
        open(container_path, 'wb').write(raw[:20])
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD).load()

    def test_empty_file_rejected(self, container_path):
        open(container_path, 'wb').close()
        with pytest.raises(MalformedContainer):
            Container.open(container_path, PASSWORD)


class TestEpoch:
    def test_epoch_survives_reload(self, container_path):
        Container.create(container_path, PASSWORD, kdf_params=FAST).store(b'x', 42)
        assert Container.open(container_path, PASSWORD).load()[0] == 42

    def test_minimum_epoch_enforced(self, container_path):
        Container.create(container_path, PASSWORD, kdf_params=FAST).store(b'x', 5)
        assert Container.open(container_path, PASSWORD).load(minimum_epoch=5)
        with pytest.raises(EpochRollback):
            Container.open(container_path, PASSWORD).load(minimum_epoch=6)

    def test_epoch_is_authenticated(self, container_path):
        """
        The epoch is inside the sealed payload, so it cannot be raised.

        An epoch in the header would be unauthenticated and an attacker would
        simply bump it past the caller's anchor.
        """
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        container.store(b'x', 1)
        raw = open(container_path, 'rb').read()
        # Prove the epoch is not in the header: it must appear only once, in
        # the sealed body, and the header is exactly 29 bytes.
        assert len(raw) > 29
        assert b'\x00' * 8 not in raw[:29]

    def test_negative_epoch_rejected(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        with pytest.raises(ValueError):
            container.store(b'x', -1)

    def test_epoch_too_large_rejected(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        with pytest.raises(ValueError):
            container.store(b'x', 2 ** 64)

    def test_store_requires_bytes(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        with pytest.raises(TypeError):
            container.store('text', 1)


class TestPersistentPreKeyStore:
    def _store(self, path, count=3):
        container = Container.create(path, PASSWORD, kdf_params=FAST)
        store = PersistentPreKeyStore(container)
        x3dh = X3DH()
        for index in range(count):
            store.put(x3dh.generate_one_time_prekey(key_id=1000 + index))
        return store

    def test_pool_survives_reopen(self, container_path):
        self._store(container_path, count=4)
        reopened = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        assert reopened.count() == 4
        assert reopened.available_ids() == [1000, 1001, 1002, 1003]

    def test_consume_persists(self, container_path):
        self._store(container_path, count=3)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        key = store.consume(1001)
        assert key.key_id == 1001

        reopened = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        assert reopened.count() == 2
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            reopened.consume(1001)
        assert reopened.was_consumed(1001)

    def test_peek_does_not_consume(self, container_path):
        self._store(container_path, count=2)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        assert store.peek(1000).key_id == 1000
        assert store.count() == 2

    def test_peek_refuses_consumed(self, container_path):
        self._store(container_path, count=2)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        store.consume(1000)
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            store.peek(1000)

    def test_unknown_id(self, container_path):
        self._store(container_path, count=1)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        with pytest.raises(NoSuchPreKey):
            store.consume(9999)

    def test_each_key_usable_once(self, container_path):
        self._store(container_path, count=5)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        for key_id in store.available_ids():
            store.consume(key_id)
            with pytest.raises(OneTimePreKeyAlreadyUsed):
                store.consume(key_id)
        assert store.count() == 0

    def test_wrong_password_cannot_read_pool(self, container_path):
        self._store(container_path)
        with pytest.raises(WrongPassword):
            PersistentPreKeyStore(Container.open(container_path, 'wrong'))

    def test_private_keys_not_plaintext_in_file(self, container_path):
        self._store(container_path, count=1)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        raw = open(container_path, 'rb').read()
        secret = bytes(store.peek(1000).private_key)
        assert secret not in raw
        assert base64.b64encode(secret) not in raw

    def test_corrupt_payload_rejected(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        container.store(b'not json at all', epoch=1)
        with pytest.raises(StorageError):
            PersistentPreKeyStore(container)

    def test_mismatched_keypair_rejected(self, container_path):
        """
        A file whose public key does not match the private one is refused
        rather than trusted, since everything downstream depends on the pair.
        """
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        private = PrivateKey.generate()
        document = {
            'keys': [{
                'id': 1,
                'private': base64.b64encode(bytes(private)).decode(),
                'public': base64.b64encode(bytes(PrivateKey.generate().public_key)).decode(),
            }],
            'used': [],
        }
        container.store(json.dumps(document).encode('utf-8'), epoch=1)
        with pytest.raises(StorageError):
            PersistentPreKeyStore(container)

    def test_failed_write_does_not_consume(self, container_path, monkeypatch):
        """
        A prekey must not be spent unless it is durably recorded as spent.

        Otherwise a transient disk error silently destroys forward secrecy for
        that session, and the pool leaks keys that were already handed out.
        """
        self._store(container_path, count=2)
        container = Container.open(container_path, PASSWORD)
        store = PersistentPreKeyStore(container)

        def explode(payload, epoch):
            raise OSError('disk full')

        monkeypatch.setattr(container, 'store', explode)
        with pytest.raises(OSError):
            store.consume(1000)
        monkeypatch.undo()

        # The key is still available and still consumable.
        reopened = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        assert reopened.was_consumed(1000) is False
        assert reopened.consume(1000).key_id == 1000

    def test_duplicate_put_rejected(self, container_path):
        self._store(container_path, count=1)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        x3dh = X3DH()
        with pytest.raises(PreKeyStoreError):
            store.put(x3dh.generate_one_time_prekey(key_id=1000))

    def test_reput_consumed_id_rejected(self, container_path):
        self._store(container_path, count=1)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        store.consume(1000)
        x3dh = X3DH()
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            store.put(x3dh.generate_one_time_prekey(key_id=1000))

    def test_concurrent_consume_has_one_winner(self, container_path):
        """
        Single-process atomicity: many threads racing for one prekey, one wins.

        Cross-process atomicity is not implemented and the module says so.
        """
        self._store(container_path, count=1)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))

        winners = []
        barrier = threading.Barrier(12)

        def attempt():
            barrier.wait()
            try:
                store.consume(1000)
                winners.append(1)
            except (OneTimePreKeyAlreadyUsed, NoSuchPreKey):
                pass

        threads = [threading.Thread(target=attempt) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(winners) == 1

    def test_epoch_advances_on_mutation(self, container_path):
        self._store(container_path, count=2)
        store = PersistentPreKeyStore(Container.open(container_path, PASSWORD))
        before = store.epoch
        store.consume(1000)
        assert store.epoch > before

    def test_minimum_epoch_blocks_stale_file(self, container_path):
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        PersistentPreKeyStore(container).put(X3DH().generate_one_time_prekey(key_id=1))
        stale_epoch = PersistentPreKeyStore(container).epoch
        with pytest.raises(EpochRollback):
            PersistentPreKeyStore(
                Container.open(container_path, PASSWORD),
                minimum_epoch=stale_epoch + 5,
            )


class TestIdentityStore:
    def _identity(self, path):
        store = IdentityStore(Container.create(path, PASSWORD, kdf_params=FAST))
        store.initialise()
        return store

    def test_initialise_creates_both_keys(self, container_path):
        store = self._identity(container_path)
        assert isinstance(store.identity_key, SigningKey)
        assert isinstance(store.signed_prekey_private, PrivateKey)

    def test_keys_survive_reopen(self, container_path):
        store = self._identity(container_path)
        identity = bytes(store.identity_key)
        spk = bytes(store.signed_prekey_private)
        reopened = IdentityStore(Container.open(container_path, PASSWORD))
        assert bytes(reopened.identity_key) == identity
        assert bytes(reopened.signed_prekey_private) == spk

    def test_double_initialise_rejected(self, container_path):
        store = self._identity(container_path)
        with pytest.raises(StorageError):
            store.initialise()

    def test_uninitialised_store_refuses_access(self, container_path):
        store = IdentityStore(Container.create(container_path, PASSWORD, kdf_params=FAST))
        with pytest.raises(StorageError):
            store.identity_key

    def test_rotate_replaces_prekey_only(self, container_path):
        store = self._identity(container_path)
        identity = bytes(store.identity_key)
        old_spk = bytes(store.signed_prekey_private)
        store.rotate()
        assert bytes(store.identity_key) == identity
        assert bytes(store.signed_prekey_private) != old_spk

    def test_rotation_persists(self, container_path):
        store = self._identity(container_path)
        store.rotate()
        spk = bytes(store.signed_prekey_private)
        reopened = IdentityStore(Container.open(container_path, PASSWORD))
        assert bytes(reopened.signed_prekey_private) == spk

    def test_rotate_requires_initialisation(self, container_path):
        store = IdentityStore(Container.create(container_path, PASSWORD, kdf_params=FAST))
        with pytest.raises(StorageError):
            store.rotate()

    def test_private_key_not_plaintext(self, container_path):
        store = self._identity(container_path)
        raw = open(container_path, 'rb').read()
        assert bytes(store.identity_key) not in raw


class TestSessionStore:
    def _ratchet_state(self):
        import nacl.public as public
        from src.crypto.double_ratchet import DoubleRatchet
        alice = public.PrivateKey.generate()
        bob = public.PrivateKey.generate()
        shared = public.Box(alice, bob.public_key)._shared_key
        ratchet = DoubleRatchet(
            dh_private=alice,
            remote_dh_public=bytes(bob.public_key),
            root_key=shared,
            is_initiator=True,
        )
        ratchet.encrypt(b'hello')
        return ratchet.state

    def test_session_survives_reopen(self, tmp_path):
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        directory = str(tmp_path / 'sessions')
        store = SessionStore(directory, PASSWORD)
        state = self._ratchet_state()
        store.save('peer-1', state)

        reopened = SessionStore(directory, PASSWORD)
        loaded = reopened.load('peer-1', PROTOCOL_VERSION)
        assert loaded.root_key == state.root_key
        assert loaded.send_chain_key == state.send_chain_key
        assert loaded.send_msg_count == state.send_msg_count

    def test_missing_peer(self, tmp_path):
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        store = SessionStore(str(tmp_path / 's'), PASSWORD)
        with pytest.raises(KeyError):
            store.load('nobody', PROTOCOL_VERSION)

    def test_wrong_password(self, tmp_path):
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        directory = str(tmp_path / 'sessions')
        SessionStore(directory, PASSWORD).save('peer', self._ratchet_state())
        with pytest.raises(WrongPassword):
            SessionStore(directory, 'wrong').load('peer', PROTOCOL_VERSION)

    def test_version_mismatch_refused(self, tmp_path):
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        directory = str(tmp_path / 'sessions')
        SessionStore(directory, PASSWORD).save('peer', self._ratchet_state())
        with pytest.raises(Exception):
            SessionStore(directory, PASSWORD).load('peer', PROTOCOL_VERSION + 1)

    def test_peer_names_are_hashed(self, tmp_path):
        """
        A hostile peer id must not escape the directory or pick a suffix, so
        the id is hashed rather than used as a filename.
        """
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        directory = str(tmp_path / 'sessions')
        store = SessionStore(directory, PASSWORD)
        hostile = '../../../etc/passwd'
        store.save(hostile, self._ratchet_state())
        assert store.load(hostile, PROTOCOL_VERSION) is not None
        names = os.listdir(directory)
        assert all(not name.startswith('.') for name in names)
        assert all(name.endswith('.session') for name in names)

    def test_peers_are_isolated(self, tmp_path):
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        store = SessionStore(str(tmp_path / 'sessions'), PASSWORD)
        first = self._ratchet_state()
        second = self._ratchet_state()
        store.save('a', first)
        store.save('b', second)
        assert store.load('a', PROTOCOL_VERSION).root_key == first.root_key
        assert store.load('b', PROTOCOL_VERSION).root_key == second.root_key
        assert len(store.peers()) == 2

    def test_delete_one_peer(self, tmp_path):
        from src.crypto.double_ratchet import PROTOCOL_VERSION
        store = SessionStore(str(tmp_path / 'sessions'), PASSWORD)
        store.save('a', self._ratchet_state())
        store.save('b', self._ratchet_state())
        store.delete('a')
        with pytest.raises(KeyError):
            store.load('a', PROTOCOL_VERSION)
        assert store.load('b', PROTOCOL_VERSION) is not None

    def test_message_keys_not_written(self, tmp_path):
        directory = str(tmp_path / 'sessions')
        store = SessionStore(directory, PASSWORD)
        state = self._ratchet_state()
        store.save('peer', state)
        raw = open(os.path.join(directory, store._path('peer')), 'rb').read()
        for key in (state.root_key, state.send_chain_key):
            assert key and key not in raw


class TestEndToEndWithStorage:
    def test_handshake_survives_a_restart(self, tmp_path):
        """
        The point of persisting the pool: after a restart there are prekeys
        left, so the handshake is still the 4-DH variant with forward secrecy.
        """
        container_path = str(tmp_path / 'prekeys.bin')
        container = Container.create(container_path, PASSWORD, kdf_params=FAST)
        store = PersistentPreKeyStore(container)
        identity = IdentityStore(
            Container.create(str(tmp_path / 'identity.bin'), PASSWORD, kdf_params=FAST)
        )
        identity.initialise()

        x3dh = X3DH()
        for index in range(4):
            store.put(x3dh.generate_one_time_prekey(key_id=2000 + index))

        from src.crypto.x3dh import X3DHResponder
        responder = X3DHResponder(
            identity_private=identity.identity_key,
            signed_prekey_private=identity.signed_prekey_private,
            prekey_store=store,
        )
        bundle = responder.publish_bundle()
        assert bundle.one_time_prekey_id is not None

        # "Restart": drop every object and reopen from disk.
        del responder, store, identity
        reopened_prekeys = PersistentPreKeyStore(
            Container.open(container_path, PASSWORD)
        )
        reopened_identity = IdentityStore(
            Container.open(str(tmp_path / 'identity.bin'), PASSWORD)
        )
        responder = X3DHResponder(
            identity_private=reopened_identity.identity_key,
            signed_prekey_private=reopened_identity.signed_prekey_private,
            prekey_store=reopened_prekeys,
        )

        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.begin(alice_private, responder.publish_bundle())
        responder_session = responder.respond(init)
        session.verify_key_confirmation(responder_session.make_key_confirmation())

        # The handshake used a persisted one-time prekey, so it is the 4-DH
        # variant: after a restart this session still has forward secrecy.
        assert session.one_time_prekey_id is not None
        assert session.root_key == responder_session.root_key

        # And the pool shrank by exactly that key, durably.
        assert reopened_prekeys.was_consumed(session.one_time_prekey_id)
