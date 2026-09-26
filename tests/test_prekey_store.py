"""
Tests for one-time prekey single-use enforcement.

Reusing a one-time prekey gives an attacker who observed both sessions enough
material to recover the earlier shared secret, so single use is a security
property of the store, not a convention in the caller.
"""

import threading

import pytest

from src.crypto.prekey_store import (
    InMemoryPreKeyStore,
    NoSuchPreKey,
    OneTimePreKey,
    OneTimePreKeyAlreadyUsed,
    PreKeyStore,
    generate_prekey_id,
    make_one_time_prekey,
)


class TestConsume:
    def test_consume_returns_the_key(self):
        store = InMemoryPreKeyStore()
        key = make_one_time_prekey(key_id=1)
        store.put(key)
        assert store.consume(1) is key

    def test_second_consume_is_refused(self):
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=1))
        store.consume(1)
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            store.consume(1)

    def test_unknown_id_raises(self):
        store = InMemoryPreKeyStore()
        with pytest.raises(NoSuchPreKey):
            store.consume(404)

    def test_consume_removes_from_pool(self):
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=1))
        store.put(make_one_time_prekey(key_id=2))
        assert store.count() == 2
        store.consume(1)
        assert store.count() == 1
        assert store.available_ids() == [2]

    def test_reputting_a_consumed_id_is_refused(self):
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=1))
        store.consume(1)
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            store.put(make_one_time_prekey(key_id=1))

    def test_each_key_is_usable_exactly_once_in_a_loop(self):
        store = InMemoryPreKeyStore()
        keys = [make_one_time_prekey(key_id=index) for index in range(10)]
        store.bulk_put(keys)

        for key in keys:
            assert store.consume(key.key_id).key_id == key.key_id
            with pytest.raises(OneTimePreKeyAlreadyUsed):
                store.consume(key.key_id)
        assert store.count() == 0


class TestPeek:
    def test_peek_does_not_consume(self):
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=1))
        assert store.peek(1).key_id == 1
        assert store.peek(1).key_id == 1
        assert store.count() == 1

    def test_peek_refuses_consumed_key(self):
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=1))
        store.consume(1)
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            store.peek(1)

    def test_peek_any_returns_none_when_empty(self):
        assert InMemoryPreKeyStore().peek_any() is None

    def test_peek_any_does_not_consume(self):
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=5))
        assert store.peek_any().key_id == 5
        assert store.count() == 1


class TestConcurrency:
    def test_consume_is_atomic_under_contention(self):
        """
        Many threads racing for one prekey: exactly one may win.

        This is the property a database-backed store has to reproduce, and the
        reason ``consume`` is a single call rather than get-then-delete.
        """
        store = InMemoryPreKeyStore()
        store.put(make_one_time_prekey(key_id=99))

        winners = []
        barrier = threading.Barrier(16)

        def attempt():
            barrier.wait()
            try:
                store.consume(99)
                winners.append(1)
            except (OneTimePreKeyAlreadyUsed, NoSuchPreKey):
                pass

        threads = [threading.Thread(target=attempt) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(winners) == 1


class TestValueObject:
    def test_rejects_negative_id(self):
        with pytest.raises(ValueError):
            OneTimePreKey(
                key_id=-1,
                private_key=PrivateKeyFactory(),
                public_key=PrivateKeyFactory().public_key,
            )

    def test_rejects_oversized_public_key(self):
        private = make_one_time_prekey(key_id=1)
        with pytest.raises(ValueError):
            OneTimePreKey(
                key_id=1,
                private_key=private.private_key,
                public_key=b'\x00' * 31,  # type: ignore[arg-type]
            )

    def test_generated_ids_are_distinct(self):
        ids = {generate_prekey_id() for _ in range(200)}
        assert len(ids) == 200

    def test_store_is_abstract(self):
        with pytest.raises(TypeError):
            PreKeyStore()  # type: ignore[abstract]


def PrivateKeyFactory():  # noqa: N802 - small helper to keep the test readable
    from nacl.public import PrivateKey
    return PrivateKey.generate()
