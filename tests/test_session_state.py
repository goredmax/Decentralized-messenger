"""
Tests for reopening a Double Ratchet session from stored state.

Session state is persisted between processes, so a ratchet has to be
reconstructible from it. Two properties matter and are easy to lose:

- a reopened session continues the chain, rather than starting a new one. A
  ratchet step performed on load would advance the root key and desynchronise
  the peer permanently.
- a session reopened from a state that omitted skipped message keys cannot
  decrypt outstanding messages, and says so, rather than failing in some
  surprising way later.
"""

import json

import pytest
import nacl.public as public

from src.crypto.double_ratchet import (
    DoubleRatchet,
    IncompatibleSessionError,
    InvalidSessionStateError,
    PROTOCOL_VERSION,
    SessionState,
)


def create_session_pair():
    alice_dh = public.PrivateKey.generate()
    bob_dh = public.PrivateKey.generate()
    shared_secret = public.Box(alice_dh, bob_dh.public_key)._shared_key
    alice = DoubleRatchet(
        dh_private=alice_dh,
        remote_dh_public=bytes(bob_dh.public_key),
        root_key=shared_secret,
        is_initiator=True,
    )
    bob = DoubleRatchet(
        dh_private=bob_dh,
        remote_dh_public=None,
        root_key=shared_secret,
        is_initiator=False,
    )
    return alice, bob


class TestReopen:
    def test_round_trip_preserves_position(self):
        alice, _ = create_session_pair()
        for index in range(3):
            alice.encrypt(f"Msg {index}".encode())
        reopened = DoubleRatchet.from_state(
            SessionState.deserialize(alice.state.serialize())
        )
        assert reopened.state.root_key == alice.state.root_key
        assert reopened.state.send_chain_key == alice.state.send_chain_key
        assert reopened.state.send_msg_count == alice.state.send_msg_count
        assert reopened.state.dh_local_priv == alice.state.dh_local_priv

    def test_no_ratchet_step_on_load(self):
        """
        Loading must not advance the root key.

        If it did, the peer would never derive the same key again and the
        session would die at the next message.
        """
        alice, _ = create_session_pair()
        before = alice.state.root_key
        DoubleRatchet.from_state(SessionState.deserialize(alice.state.serialize()))
        assert alice.state.root_key == before

    def test_continues_an_existing_conversation(self):
        """The real test: reopen mid-conversation and carry on."""
        alice, bob = create_session_pair()

        first, first_hdr = alice.encrypt(b"before restart")
        assert bob.decrypt(first, first_hdr) == b"before restart"

        # Both sides persist and reload.
        alice = DoubleRatchet.from_state(
            SessionState.deserialize(alice.state.serialize())
        )
        bob = DoubleRatchet.from_state(
            SessionState.deserialize(bob.state.serialize())
        )

        second, second_hdr = alice.encrypt(b"after restart")
        assert bob.decrypt(second, second_hdr) == b"after restart"

        # And a reply, which triggers a DH ratchet on the reloaded ratchets.
        reply, reply_hdr = bob.encrypt(b"reply after restart")
        assert alice.decrypt(reply, reply_hdr) == b"reply after restart"

        final, final_hdr = alice.encrypt(b"still working")
        assert bob.decrypt(final, final_hdr) == b"still working"

    def test_many_restarts(self):
        alice, bob = create_session_pair()
        for round_index in range(6):
            message = f"round {round_index}".encode()
            encrypted, header = alice.encrypt(message)
            assert bob.decrypt(encrypted, header) == message

            alice = DoubleRatchet.from_state(
                SessionState.deserialize(alice.state.serialize())
            )
            bob = DoubleRatchet.from_state(
                SessionState.deserialize(bob.state.serialize())
            )

    def test_state_survives_json(self):
        """The state really is JSON, so a real store can hold it."""
        alice, _ = create_session_pair()
        alice.encrypt(b"x")
        payload = json.dumps(alice.state.serialize())
        reopened = DoubleRatchet.from_state(
            SessionState.deserialize(json.loads(payload))
        )
        assert reopened.state.root_key == alice.state.root_key

    def test_rejects_non_state(self):
        with pytest.raises(Exception):
            DoubleRatchet.from_state(object())  # type: ignore[arg-type]

    def test_rejects_a_foreign_version(self):
        alice, _ = create_session_pair()
        payload = alice.state.serialize()
        payload['version'] = PROTOCOL_VERSION + 1
        with pytest.raises(IncompatibleSessionError):
            DoubleRatchet.from_state(SessionState.deserialize(payload))

    def test_rejects_a_dict_without_a_version(self):
        """from_state takes a serialized dict, so a bare dict is refused."""
        with pytest.raises(IncompatibleSessionError):
            DoubleRatchet.from_state({'root_key': '00' * 32})

    def test_rejects_a_malformed_dict(self):
        payload = create_session_pair()[0].state.serialize()
        del payload['dh_local_priv']
        with pytest.raises(Exception):
            DoubleRatchet.from_state(payload)

    def test_inconsistent_key_pair_refused_eagerly(self):
        """
        dh_local_pub is derivable from dh_local_priv, so a mismatch means the
        file was rewritten rather than merely truncated.

        The check runs in __post_init__, so the state cannot even be built in a
        tampered shape, let alone loaded from disk.
        """
        alice, _ = create_session_pair()
        state = alice.state
        with pytest.raises(InvalidSessionStateError):
            SessionState(
                dh_local_priv=state.dh_local_priv,
                dh_local_pub=public.PrivateKey.generate().public_key.__bytes__(),
                dh_remote_pub=state.dh_remote_pub,
                root_key=state.root_key,
                send_chain_key=state.send_chain_key,
                recv_chain_key=state.recv_chain_key,
                send_msg_count=state.send_msg_count,
                recv_msg_count=state.recv_msg_count,
                prev_send_count=state.prev_send_count,
                skipped_keys=state.skipped_keys,
            )

    def test_wrong_length_root_key_refused(self):
        alice, _ = create_session_pair()
        state = alice.state
        with pytest.raises(InvalidSessionStateError):
            SessionState(
                dh_local_priv=state.dh_local_priv,
                dh_local_pub=state.dh_local_pub,
                dh_remote_pub=state.dh_remote_pub,
                root_key=b'\x00' * 16,
                send_chain_key=state.send_chain_key,
                recv_chain_key=state.recv_chain_key,
            )

    def test_reopened_ratchet_copies_the_state(self):
        """Mutating the caller's object must not reach a live session."""
        alice, _ = create_session_pair()
        snapshot = SessionState.deserialize(alice.state.serialize())
        reopened = DoubleRatchet.from_state(snapshot)
        snapshot.send_msg_count = 999
        assert reopened.state.send_msg_count != 999


class TestOutstandingMessagesAfterReopen:
    def _alice_sends_three(self):
        """
        Alice sends three; Bob receives only the last.

        That is what creates skipped keys: the first two message keys are
        derived and retained so the delayed messages stay decryptable.
        """
        alice, bob = create_session_pair()
        held = [alice.encrypt(f"Msg {index}".encode()) for index in range(3)]
        bob.decrypt(*held[2])
        return alice, bob, held

    def test_in_flight_messages_become_undecryptable(self):
        """
        The documented trade-off, made explicit.

        A message that was skipped over is decrypted from a retained message
        key. Those keys are not written by default, so after a restart the
        message is simply gone. That is preferable to persisting them, but it
        must be a decision, not a surprise.
        """
        _, bob, held = self._alice_sends_three()
        assert len(bob.state.skipped_keys) == 2

        bob = DoubleRatchet.from_state(
            SessionState.deserialize(bob.state.serialize())
        )
        assert len(bob.state.skipped_keys) == 0

        with pytest.raises(Exception):
            bob.decrypt(*held[0])

    def test_opting_in_preserves_in_flight_messages(self):
        """
        include_message_keys=True keeps them working, and the test states the
        cost so a reader cannot miss it.
        """
        _, bob, held = self._alice_sends_three()
        payload = json.dumps(bob.state.serialize(include_message_keys=True))
        bob = DoubleRatchet.from_state(
            SessionState.deserialize(json.loads(payload))
        )
        assert len(bob.state.skipped_keys) == 2
        assert bob.decrypt(*held[0]) == b"Msg 0"
        assert bob.decrypt(*held[1]) == b"Msg 1"

    def test_message_keys_would_be_exposed_on_disk(self):
        """Why the default is what it is."""
        _, bob, _ = self._alice_sends_three()
        assert bob.state.skipped_keys

        leaky = json.dumps(bob.state.serialize(include_message_keys=True))
        for key in bob.state.skipped_keys.values():
            assert key.hex() in leaky, 'this is the exposure being documented'
