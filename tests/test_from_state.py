"""
Tests for restoring a Double Ratchet from persisted state.

:func:`DoubleRatchet.from_state` is the boundary where bytes from disk become a
live key schedule, so the tests here are split in two: does a restored session
actually keep working, and is a damaged one refused loudly rather than loaded.

The second half matters more than the first. A restored session that decrypts
the *wrong* thing is worse than one that refuses to start, and the most likely
way to get there is a state file whose ``dh_local_pub`` no longer belongs to its
``dh_local_priv`` — the two are stored separately but describe one key.
"""

import json
from collections import OrderedDict

import nacl.public as public
import pytest

from src.crypto.double_ratchet import (
    MAX_SKIP,
    MAX_STORED_SKIPPED,
    PROTOCOL_VERSION,
    DoubleRatchet,
    DuplicateMessageError,
    IncompatibleSessionError,
    InvalidSessionStateError,
    TooManySkippedMessagesError,
)

#: The three counters, bounded identically by SessionState.validate.
COUNTERS = ['send_msg_count', 'recv_msg_count', 'prev_send_count']


def create_session_pair():
    """A ratchet pair sharing a root key, without going through X3DH."""
    alice_dh = public.PrivateKey.generate()
    bob_dh = public.PrivateKey.generate()
    root_key = public.Box(alice_dh, bob_dh.public_key)._shared_key

    alice = DoubleRatchet(
        dh_private=alice_dh,
        remote_dh_public=bytes(bob_dh.public_key),
        root_key=root_key,
        is_initiator=True,
    )
    bob = DoubleRatchet(
        dh_private=bob_dh,
        remote_dh_public=None,
        root_key=root_key,
        is_initiator=False,
    )
    return alice, bob


def tamper(payload, **changes):
    """A serialized state dict with fields replaced, for corruption tests."""
    damaged = json.loads(json.dumps(payload))
    damaged.update(changes)
    return damaged


class TestRestoreContinuesSession:
    def test_restored_initiator_can_still_send_and_be_understood(self):
        """The basic contract: save mid-conversation, reload, keep talking."""
        alice, bob = create_session_pair()
        for i in range(3):
            enc, hdr = alice.encrypt(f"before {i}".encode())
            assert bob.decrypt(enc, hdr) == f"before {i}".encode()

        restored = DoubleRatchet.from_state(alice.state.serialize())

        for i in range(3):
            enc, hdr = restored.encrypt(f"after {i}".encode())
            assert bob.decrypt(enc, hdr) == f"after {i}".encode()

    def test_restored_responder_can_still_receive_and_reply(self):
        alice, bob = create_session_pair()
        enc, hdr = alice.encrypt(b"first")
        bob.decrypt(enc, hdr)

        restored = DoubleRatchet.from_state(bob.state.serialize())

        enc, hdr = alice.encrypt(b"second")
        assert restored.decrypt(enc, hdr) == b"second"

        reply, reply_hdr = restored.encrypt(b"reply")
        assert alice.decrypt(reply, reply_hdr) == b"reply"

    def test_restore_does_not_ratchet(self):
        """
        ``__init__`` ratchets for the initiator. Restoring through it would
        advance the root key and throw away the sending chain: the session would
        look fine locally and fail at the peer on the very next message.
        """
        alice, _bob = create_session_pair()
        before = alice.state.serialize()
        restored = DoubleRatchet.from_state(before)

        assert restored.state.root_key == bytes.fromhex(before['root_key'])
        assert restored.state.send_chain_key == bytes.fromhex(before['send_chain_key'])
        assert restored.state.dh_local_pub == bytes.fromhex(before['dh_local_pub'])
        assert restored.state.dh_local_priv == bytes.fromhex(before['dh_local_priv'])

    def test_round_trip_through_several_dh_ratchets(self):
        """Both sides reload repeatedly, across a chain of DH ratchets."""
        alice, bob = create_session_pair()

        for turn in range(4):
            enc, hdr = alice.encrypt(f"a{turn}".encode())
            bob.decrypt(enc, hdr)
            enc, hdr = bob.encrypt(f"b{turn}".encode())
            alice.decrypt(enc, hdr)

            alice = DoubleRatchet.from_state(alice.state.serialize())
            bob = DoubleRatchet.from_state(bob.state.serialize())

        enc, hdr = alice.encrypt(b"final")
        assert bob.decrypt(enc, hdr) == b"final"

    def test_restored_responder_state_with_no_chains_yet(self):
        """
        A responder that has not heard from its peer has no chains at all.
        That is a legitimate state, not a half-built one, and must load — and
        must still be waiting for the first message.
        """
        alice, bob = create_session_pair()
        assert bob.state.send_chain_key is None
        assert bob.state.recv_chain_key is None
        assert bob.state.dh_remote_pub is None

        restored = DoubleRatchet.from_state(bob.state.serialize())
        assert restored.state.send_chain_key is None
        assert restored.state.dh_remote_pub is None

        # It can receive, and it ratchets into a sending chain on the way.
        assert restored.decrypt(*alice.encrypt(b"first contact")) == b"first contact"
        assert restored.state.dh_remote_pub is not None
        assert restored.state.send_chain_key is not None

    def test_restored_responder_with_no_peer_key_cannot_send(self):
        """
        A responder that has never heard from its peer has no key to ratchet
        to, and must say so rather than derive a chain from nothing.
        """
        _alice, bob = create_session_pair()
        restored = DoubleRatchet.from_state(bob.state.serialize())

        with pytest.raises(InvalidSessionStateError, match='remote DH public key'):
            restored.encrypt(b"unsolicited")

    def test_accepts_a_session_state_object(self):
        """SessionStore.load returns a SessionState, so that must be accepted."""
        alice, _bob = create_session_pair()
        alice.encrypt(b"warm up")

        restored = DoubleRatchet.from_state(alice.state)

        assert restored.state.root_key == alice.state.root_key
        assert restored.state.send_msg_count == alice.state.send_msg_count

    def test_restored_ratchet_does_not_alias_the_caller_state(self):
        """
        The ratchet must own its state.

        If it aliased the caller's object, writing the session back out would
        mutate a live session through the ``state`` property, and rekeying would
        be a silent, unsupported side effect of a storage call.
        """
        alice, _bob = create_session_pair()
        snapshot = alice.state
        restored = DoubleRatchet.from_state(snapshot)

        assert restored.state is not snapshot
        assert restored.state.skipped_keys is not snapshot.skipped_keys

        snapshot.root_key = b'\x00' * 32
        assert restored.state.root_key != b'\x00' * 32


class TestSkippedKeysAcrossRestore:
    def test_out_of_order_survives_when_message_keys_are_kept(self):
        """
        A message that arrived out of order is decryptable from the retained
        key, so that key has to be part of the restored state.
        """
        alice, bob = create_session_pair()
        messages = [f"msg {i}".encode() for i in range(5)]
        sent = [alice.encrypt(m) for m in messages]

        # Deliver 0, 1 and 4: keys for 2 and 3 are retained, and 4 forces a skip.
        for index in (0, 1, 4):
            bob.decrypt(*sent[index])

        restored = DoubleRatchet.from_state(
            bob.state.serialize(include_message_keys=True)
        )

        for index in (2, 3):
            assert restored.decrypt(*sent[index]) == messages[index]

    def test_out_of_order_is_undecryptable_after_a_default_save(self):
        """
        The documented trade-off, asserted so it stays a decision.

        ``serialize()`` drops message keys, so a reload without them cannot
        decrypt what is still in flight. That is deliberate: persisting a
        message key next to the session turns a storage compromise into a
        decryption capability. A caller who needs the in-flight messages must
        ask for them and accept the cost.
        """
        alice, bob = create_session_pair()
        messages = [f"msg {i}".encode() for i in range(4)]
        sent = [alice.encrypt(m) for m in messages]
        for index in (0, 3):
            bob.decrypt(*sent[index])

        restored = DoubleRatchet.from_state(bob.state.serialize())

        assert 'skipped_keys' not in bob.state.serialize()
        assert restored.state.skipped_keys == OrderedDict()
        # Without the retained key the message is not merely mis-decrypted: the
        # chain has already moved past it, so it is refused as a duplicate.
        with pytest.raises(DuplicateMessageError):
            restored.decrypt(*sent[2])


class TestDamagedStateIsRefused:
    def test_public_key_not_matching_private_key(self):
        """
        The load-bearing check.

        ``dh_local_pub`` is what goes in every message header and
        ``dh_local_priv`` is what performs the DH. If they disagree, encryption
        still succeeds locally and the peer fails to derive our message key, so
        every message is lost with no local error to explain why.
        """
        alice, _bob = create_session_pair()
        impostor = public.PrivateKey.generate()
        payload = tamper(
            alice.state.serialize(),
            dh_local_pub=bytes(impostor.public_key).hex(),
        )

        with pytest.raises(InvalidSessionStateError, match='dh_local_pub'):
            DoubleRatchet.from_state(payload)

    def test_missing_public_key_field(self):
        alice, _bob = create_session_pair()
        payload = alice.state.serialize()
        del payload['dh_local_pub']

        with pytest.raises(InvalidSessionStateError):
            DoubleRatchet.from_state(payload)

    def test_truncated_private_key(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), dh_local_priv='00' * 31)

        with pytest.raises(InvalidSessionStateError, match='dh_local_priv'):
            DoubleRatchet.from_state(payload)

    def test_wrong_length_root_key(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), root_key='00' * 16)

        with pytest.raises(InvalidSessionStateError, match='root_key'):
            DoubleRatchet.from_state(payload)

    def test_wrong_length_chain_key(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), send_chain_key='00' * 31)

        with pytest.raises(InvalidSessionStateError, match='send_chain_key'):
            DoubleRatchet.from_state(payload)

    def test_wrong_length_remote_public_key(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), dh_remote_pub='00' * 31)

        with pytest.raises(InvalidSessionStateError, match='dh_remote_pub'):
            DoubleRatchet.from_state(payload)

    def test_non_hex_field(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), root_key='not hex at all')

        with pytest.raises(InvalidSessionStateError, match='hex'):
            DoubleRatchet.from_state(payload)

    def test_field_of_wrong_type(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), root_key=12345)

        with pytest.raises(InvalidSessionStateError):
            DoubleRatchet.from_state(payload)

    @pytest.mark.parametrize('counter', COUNTERS)
    def test_negative_counter(self, counter):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), **{counter: -1})

        with pytest.raises(InvalidSessionStateError, match=counter):
            DoubleRatchet.from_state(payload)

    @pytest.mark.parametrize('counter', COUNTERS)
    def test_counter_beyond_the_wire_range(self, counter):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), **{counter: 2 ** 32})

        with pytest.raises(InvalidSessionStateError, match=counter):
            DoubleRatchet.from_state(payload)

    @pytest.mark.parametrize('counter', COUNTERS)
    def test_boolean_counter(self, counter):
        """
        ``bool`` is an ``int`` subclass, so ``True`` slips past a range check
        and then serialises as ``1``: a type confusion that would quietly become
        a wrong counter. JSON ``true`` is a realistic way in, since a state
        payload is a JSON document.
        """
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), **{counter: True})

        with pytest.raises(InvalidSessionStateError, match=counter):
            DoubleRatchet.from_state(payload)

    def test_foreign_protocol_version(self):
        alice, _bob = create_session_pair()
        payload = tamper(alice.state.serialize(), version=PROTOCOL_VERSION + 1)

        with pytest.raises(IncompatibleSessionError):
            DoubleRatchet.from_state(payload)

    def test_missing_version(self):
        alice, _bob = create_session_pair()
        payload = alice.state.serialize()
        del payload['version']

        with pytest.raises(IncompatibleSessionError):
            DoubleRatchet.from_state(payload)

    @pytest.mark.parametrize('junk', [None, 42, b'bytes', 'a string', ['a', 'list']])
    def test_junk_instead_of_state(self, junk):
        with pytest.raises(InvalidSessionStateError):
            DoubleRatchet.from_state(junk)

    def test_skipped_key_of_wrong_length(self):
        alice, bob = create_session_pair()
        sent = [alice.encrypt(f"msg {i}".encode()) for i in range(3)]
        bob.decrypt(*sent[2])
        payload = bob.state.serialize(include_message_keys=True)
        assert payload['skipped_keys']

        payload['skipped_keys'][0][2] = '00' * 16
        with pytest.raises(InvalidSessionStateError, match='skipped_keys message key'):
            DoubleRatchet.from_state(payload)

    def test_malformed_skipped_key_record(self):
        alice, bob = create_session_pair()
        sent = [alice.encrypt(f"msg {i}".encode()) for i in range(3)]
        bob.decrypt(*sent[2])
        payload = bob.state.serialize(include_message_keys=True)

        payload['skipped_keys'][0] = [payload['skipped_keys'][0][0], 1]
        with pytest.raises(InvalidSessionStateError, match='skipped_keys'):
            DoubleRatchet.from_state(payload)

    def test_more_skipped_keys_than_the_limit(self):
        """
        A state file is untrusted input here, so the store limit has to hold on
        the way in as well as on the way out. Otherwise a file can ask us to
        materialise an unbounded map of message keys.
        """
        alice, _bob = create_session_pair()
        record = [alice.state.dh_remote_pub.hex(), 0, '00' * 32]
        payload = tamper(
            alice.state.serialize(),
            skipped_keys=[record] * (MAX_STORED_SKIPPED + 1),
        )

        with pytest.raises(InvalidSessionStateError, match='MAX_STORED_SKIPPED'):
            DoubleRatchet.from_state(payload)

    def test_skipped_keys_at_the_limit_are_accepted(self):
        """
        The bound must not be off by one: a session that legitimately holds the
        maximum has to load.
        """
        alice, _bob = create_session_pair()
        records = [
            [alice.state.dh_remote_pub.hex(), i, '00' * 32]
            for i in range(MAX_STORED_SKIPPED)
        ]
        payload = tamper(alice.state.serialize(), skipped_keys=records)

        restored = DoubleRatchet.from_state(payload)
        assert len(restored.state.skipped_keys) == MAX_STORED_SKIPPED

    def test_skip_limit_still_applies_after_a_restore(self):
        """
        The budget is a property of the session, not of the process. A restored
        ratchet must still refuse a peer that tries to skip past it.
        """
        alice, bob = create_session_pair()
        for i in range(3):
            bob.decrypt(*alice.encrypt(f"msg {i}".encode()))
        restored = DoubleRatchet.from_state(bob.state.serialize())

        # The gap is measured from recv_msg_count, which is 3 by now, so the
        # header has to clear MAX_SKIP from *there* to be over the limit.
        enc, hdr = alice.encrypt(b"way ahead")
        hdr = dict(hdr, n=3 + MAX_SKIP + 1)
        with pytest.raises(TooManySkippedMessagesError):
            restored.decrypt(enc, hdr)

    def test_state_is_unchanged_after_a_refused_decrypt(self):
        """
        A rejected message must leave no trace.

        This is what keeps one hostile packet from advancing the chain: without
        the rollback the session desynchronises permanently and no later message
        can be read. Restoring from state must not weaken it.
        """
        alice, bob = create_session_pair()
        for i in range(3):
            bob.decrypt(*alice.encrypt(f"msg {i}".encode()))
        restored = DoubleRatchet.from_state(bob.state.serialize())
        before = restored.state.serialize()

        enc, hdr = alice.encrypt(b"way ahead")
        with pytest.raises(TooManySkippedMessagesError):
            restored.decrypt(enc, dict(hdr, n=3 + MAX_SKIP + 1))

        assert restored.state.serialize() == before

        # And the session is still usable afterwards.
        assert restored.decrypt(*alice.encrypt(b"still here")) == b"still here"


class TestRestoreThroughSessionStore:
    """
    The path an application actually takes.

    ``SessionStore`` hands back a ``SessionState``, which on its own cannot
    encrypt or decrypt. These exercise the whole loop — save, drop every
    in-memory object, load, carry on — because a restore that only works
    against a dict in the same process has not been shown to work at all.
    """

    def test_conversation_survives_repeated_restarts(self, tmp_path):
        from src.storage.store import SessionStore

        alice, bob = create_session_pair()
        store = SessionStore(str(tmp_path), 'correct horse battery staple')
        assert bob.decrypt(*alice.encrypt(b'before any restart')) == b'before any restart'
        store.save('bob', bob.state)

        for round_number in range(3):
            # A restart throws away every live object; only the file survives.
            del bob
            reloaded = DoubleRatchet.from_state(store.load('bob', PROTOCOL_VERSION))

            message = f'after restart {round_number}'.encode()
            assert reloaded.decrypt(*alice.encrypt(message)) == message
            store.save('bob', reloaded.state)
            bob = reloaded

    def test_tampered_state_file_is_refused_at_load(self, tmp_path):
        """
        The store authenticates its own container, so tampering is caught
        earlier. This covers the case the container cannot: a *validly* sealed
        but internally inconsistent state, which only ``validate`` can see.
        """
        from src.storage.store import SessionStore

        alice, bob = create_session_pair()
        store = SessionStore(str(tmp_path), 'pw')
        bob.decrypt(*alice.encrypt(b'warm up'))
        store.save('bob', bob.state)

        state = store.load('bob', PROTOCOL_VERSION)
        state.dh_local_pub = bytes(public.PrivateKey.generate().public_key)
        store.save('bob', state)  # the container is sealed and authenticated

        with pytest.raises(InvalidSessionStateError, match='dh_local_pub'):
            DoubleRatchet.from_state(store.load('bob', PROTOCOL_VERSION))


class TestValidationIsNotBypassable:
    def test_deserialize_also_refuses_a_mismatched_key_pair(self):
        """
        ``from_state`` is not the only door. A caller that keeps the
        ``SessionState`` and hands it to something else must not be able to
        build an unvalidated state either, so the check lives on the state
        itself.
        """
        alice, _bob = create_session_pair()
        state = alice.state
        state.dh_local_pub = bytes(public.PrivateKey.generate().public_key)

        with pytest.raises(InvalidSessionStateError, match='dh_local_pub'):
            state.validate()

    def test_validate_accepts_an_untouched_state(self):
        alice, _bob = create_session_pair()
        alice.encrypt(b"one")
        alice.state.validate()  # must not raise
