"""
Comprehensive tests for Double Ratchet implementation.
Tests cover basic functionality, security properties, and edge cases.
"""

import pytest
import nacl.public as public
from src.crypto.double_ratchet import (
    DoubleRatchet, SessionState,
    TooManySkippedMessagesError, DuplicateMessageError,
    InvalidSessionStateError, DecryptionError,
    IncompatibleSessionError, MAX_SKIP, PROTOCOL_VERSION
)


def create_session_pair():
    """Create Alice and Bob with initial X3DH-derived root key."""
    alice_dh = public.PrivateKey.generate()
    bob_dh = public.PrivateKey.generate()
    
    shared_secret = public.Box(alice_dh, bob_dh.public_key)._shared_key
    root_key = shared_secret
    
    alice = DoubleRatchet(
        dh_private=alice_dh,
        remote_dh_public=bytes(bob_dh.public_key),
        root_key=root_key,
        is_initiator=True
    )
    
    bob = DoubleRatchet(
        dh_private=bob_dh,
        remote_dh_public=None,
        root_key=root_key,
        is_initiator=False
    )
    
    return alice, bob


class TestBasicEncryption:
    def test_alice_sends_first_message(self):
        alice, bob = create_session_pair()
        plaintext = b"Hello Bob!"
        encrypted, header = alice.encrypt(plaintext)
        decrypted = bob.decrypt(encrypted, header)
        assert decrypted == plaintext
    
    def test_bob_replies(self):
        alice, bob = create_session_pair()
        
        msg1 = b"Hello Bob!"
        enc1, hdr1 = alice.encrypt(msg1)
        bob.decrypt(enc1, hdr1)
        
        msg2 = b"Hi Alice!"
        enc2, hdr2 = bob.encrypt(msg2)
        decrypted = alice.decrypt(enc2, hdr2)
        assert decrypted == msg2
    
    def test_multiple_messages_same_chain(self):
        alice, bob = create_session_pair()
        
        for i in range(10):
            msg = f"Message {i}".encode()
            enc, hdr = alice.encrypt(msg)
            dec = bob.decrypt(enc, hdr)
            assert dec == msg


class TestOutOfOrderDelivery:
    def test_receive_out_of_order(self):
        alice, bob = create_session_pair()
        
        messages = [b"First", b"Second", b"Third"]
        encrypted_msgs = []
        for msg in messages:
            enc, hdr = alice.encrypt(msg)
            encrypted_msgs.append((enc, hdr))
        
        for idx in [2, 0, 1]:
            enc, hdr = encrypted_msgs[idx]
            dec = bob.decrypt(enc, hdr)
            assert dec == messages[idx]
    
    def test_skip_message_keys_stored(self):
        alice, bob = create_session_pair()
        
        enc_msgs = []
        for i in range(5):
            enc, hdr = alice.encrypt(f"Msg {i}".encode())
            enc_msgs.append((enc, hdr))
        
        dec = bob.decrypt(*enc_msgs[4])
        assert dec == b"Msg 4"
        assert len(bob.state.skipped_keys) == 4
        
        for i in range(4):
            dec = bob.decrypt(*enc_msgs[i])
            assert dec == f"Msg {i}".encode()
        
        assert len(bob.state.skipped_keys) == 0


class TestDHRatchet:
    def test_dh_ratchet_on_reply(self):
        alice, bob = create_session_pair()
        
        alice_dh_before = alice.state.dh_local_pub
        bob_dh_before = bob.state.dh_local_pub
        
        enc1, hdr1 = alice.encrypt(b"Hello")
        bob.decrypt(enc1, hdr1)
        
        enc2, hdr2 = bob.encrypt(b"Hi")
        assert bob.state.dh_local_pub != bob_dh_before
        
        alice.decrypt(enc2, hdr2)
        assert alice.state.dh_local_pub != alice_dh_before
    
    def test_dh_ratchet_updates_remote_key(self):
        alice, bob = create_session_pair()
        
        enc1, hdr1 = alice.encrypt(b"Hello")
        bob.decrypt(enc1, hdr1)
        
        assert bob.state.dh_remote_pub == alice.state.dh_local_pub
        
        enc2, hdr2 = bob.encrypt(b"Hi")
        alice.decrypt(enc2, hdr2)
        
        assert alice.state.dh_remote_pub == bob.state.dh_local_pub


class TestPostCompromiseSecurity:
    """
    Post-Compromise Security: After a DH ratchet, an attacker with 
    old state cannot decrypt new messages.
    
    The honest test: 
    1. Attacker gets full state snapshot (RK, CKs, CKr, DHs_priv, DHr_pub, skipped_keys)
    2. Legitimate parties perform DH ratchet (both generate new ephemeral keys)
    3. Attacker tries to derive message keys from old state
    4. Attacker's derived keys differ from actual keys used after healing
    """
    
    def test_pcs_after_dh_ratchet(self):
        """
        A snapshot taken before healing must not decrypt anything sent after.

        The attacker reconstructs a ratchet from the serialised state and tries
        to decrypt post-healing traffic. It fails, because the DH ratchet
        replaced the ephemeral keys the old chain keys were bound to.
        """
        alice, bob = create_session_pair()

        first, first_hdr = alice.encrypt(b"Initial message")
        assert bob.decrypt(first, first_hdr) == b"Initial message"

        # The attacker exfiltrates Alice's state at this point.
        snapshot = SessionState.deserialize(alice.state.serialize())
        attacker = DoubleRatchet(
            dh_private=public.PrivateKey(snapshot.dh_local_priv),
            remote_dh_public=snapshot.dh_remote_pub,
            root_key=snapshot.root_key,
            is_initiator=False,
        )
        attacker.state.send_chain_key = snapshot.send_chain_key
        attacker.state.recv_chain_key = snapshot.recv_chain_key
        attacker.state.send_msg_count = snapshot.send_msg_count
        attacker.state.recv_msg_count = snapshot.recv_msg_count

        # Legitimate traffic continues; Bob's reply ratchets both sides.
        for index in range(3):
            encrypted, header = alice.encrypt(f"Message {index}".encode())
            assert bob.decrypt(encrypted, header) == f"Message {index}".encode()

        reply, reply_hdr = bob.encrypt(b"Reply after ratchet")
        assert alice.decrypt(reply, reply_hdr) == b"Reply after ratchet"

        secret = b"Secret after compromise"
        encrypted, header = alice.encrypt(secret)

        assert bob.decrypt(encrypted, header) == secret
        assert alice.state.root_key != snapshot.root_key

        # The attacker's copy of the state cannot read it.
        with pytest.raises(DecryptionError):
            attacker.decrypt(encrypted, header)

    def test_snapshot_does_not_carry_message_keys(self):
        """
        An exfiltrated state must not hand over keys for undelivered messages.

        Before this change ``serialize()`` wrote every retained message key, so
        a single state dump decrypted everything still in flight.
        """
        alice, bob = create_session_pair()
        for index in range(4):
            encrypted, header = alice.encrypt(f"Msg {index}".encode())
        bob.decrypt(encrypted, header)      # leaves three skipped keys
        assert len(bob.state.skipped_keys) == 3

        exfiltrated = bob.state.serialize()
        assert 'skipped_keys' not in exfiltrated


class TestForwardSecrecy:
    def test_old_messages_safe(self):
        alice, bob = create_session_pair()
        
        messages = []
        for i in range(5):
            msg = f"Old message {i}".encode()
            enc, hdr = alice.encrypt(msg)
            messages.append((msg, enc, hdr))
            bob.decrypt(enc, hdr)
        
        for i in range(5):
            msg = f"New message {i}".encode()
            enc, hdr = alice.encrypt(msg)
            bob.decrypt(enc, hdr)


class TestDoSProtection:
    def test_max_skip_limit(self):
        alice, bob = create_session_pair()
        
        # Send first message to establish session
        enc, hdr = alice.encrypt(b"Hello")
        bob.decrypt(enc, hdr)
        
        # Craft fake header with huge message number
        fake_header = {
            'dh': alice.state.dh_local_pub.hex(),
            'pn': 0,
            'n': MAX_SKIP + 100
        }
        
        import nacl.bindings
        import nacl.utils
        fake_mk = nacl.utils.random(32)
        nonce = nacl.utils.random(24)
        fake_ct = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            b"fake", b"", nonce, fake_mk
        )
        fake_encrypted = nonce + fake_ct
        
        with pytest.raises(TooManySkippedMessagesError):
            bob.decrypt(fake_encrypted, fake_header)


class TestDuplicateDetection:
    def test_duplicate_rejected(self):
        alice, bob = create_session_pair()
        
        enc, hdr = alice.encrypt(b"Hello")
        dec1 = bob.decrypt(enc, hdr)
        assert dec1 == b"Hello"
        
        with pytest.raises(DuplicateMessageError):
            bob.decrypt(enc, hdr)


class TestAEADIntegrity:
    def test_tampered_ciphertext_fails(self):
        alice, bob = create_session_pair()
        
        enc, hdr = alice.encrypt(b"Secret message")
        
        tampered = bytearray(enc)
        tampered[30] ^= 0xFF
        
        with pytest.raises(DecryptionError):
            bob.decrypt(bytes(tampered), hdr)
    
    def test_wrong_nonce_fails(self):
        alice, bob = create_session_pair()
        
        enc, hdr = alice.encrypt(b"Secret message")
        
        bad_enc = bytearray(enc)
        bad_enc[0] ^= 0xFF
        
        with pytest.raises(DecryptionError):
            bob.decrypt(bytes(bad_enc), hdr)


class TestSessionSerialization:
    def test_serialize_deserialize_roundtrip(self):
        alice, bob = create_session_pair()
        
        for i in range(3):
            enc, hdr = alice.encrypt(f"Msg {i}".encode())
            bob.decrypt(enc, hdr)
        
        serialized = alice.state.serialize()
        restored = SessionState.deserialize(serialized)
        
        assert restored.dh_local_priv == alice.state.dh_local_priv
        assert restored.dh_local_pub == alice.state.dh_local_pub
        assert restored.root_key == alice.state.root_key
        assert restored.send_msg_count == alice.state.send_msg_count
        assert restored.recv_msg_count == alice.state.recv_msg_count
    def test_serialized_state_is_json_safe(self):
        """
        Skipped keys used to be a dict with tuple keys, which json cannot encode.
        """
        import json
        alice, bob = create_session_pair()
        for i in range(4):
            enc, hdr = alice.encrypt(f"Msg {i}".encode())
        bob.decrypt(enc, hdr)

        dumped = json.dumps(bob.state.serialize(include_message_keys=True))
        assert json.loads(dumped)['skipped_keys']

    def test_skipped_keys_roundtrip_when_requested(self):
        alice, bob = create_session_pair()
        for i in range(4):
            enc, hdr = alice.encrypt(f"Msg {i}".encode())
        bob.decrypt(enc, hdr)

        restored = SessionState.deserialize(
            bob.state.serialize(include_message_keys=True)
        )
        assert dict(restored.skipped_keys) == dict(bob.state.skipped_keys)

    def test_versionless_state_is_rejected(self):
        """
        Pre-v2 state used a different KDF_CK order and HKDF info string.

        Loading it would silently derive the wrong keys, so it is refused.
        """
        with pytest.raises(IncompatibleSessionError):
            SessionState.deserialize({'root_key': '00' * 32})

    def test_foreign_version_is_rejected(self):
        payload = create_session_pair()[0].state.serialize()
        payload['version'] = PROTOCOL_VERSION + 1
        with pytest.raises(IncompatibleSessionError):
            SessionState.deserialize(payload)


class TestKdfOrder:
    """
    The specification defines the *next chain key* as HMAC(ck, 0x01) and the
    *message key* as HMAC(ck, 0x02). The original code had these swapped.
    Self-consistency hid the bug; it only surfaced as an incompatibility with
    libsignal.
    """

    def test_next_chain_key_uses_0x01_and_message_key_0x02(self):
        import hashlib as _hashlib
        import hmac as _hmac
        ratchet = create_session_pair()[0]
        chain_key = b'\x02' * 32
        next_chain_key, message_key = ratchet._kdf_ck(chain_key)
        assert next_chain_key == _hmac.new(
            chain_key, b'\x01', _hashlib.sha256
        ).digest()
        assert message_key == _hmac.new(
            chain_key, b'\x02', _hashlib.sha256
        ).digest()

    def test_chain_actually_advances_to_the_next_value(self):
        """The stored sending chain must be the "next" value, not the message key."""
        ratchet = create_session_pair()[0]
        assert ratchet.state.send_chain_key is not None
        before = ratchet.state.send_chain_key
        ratchet.encrypt(b'x')
        after = ratchet.state.send_chain_key
        assert after != before
        next_chain_key, _ = ratchet._kdf_ck(before)
        assert after == next_chain_key

    def test_kdf_rk_returns_two_full_length_keys(self):
        """
        Regression: the chain key used to come back empty.

        ``KDF_RK`` expanded to a single 32-byte HMAC and then read ``okm[32:64]``
        from a 32-byte buffer, so every chain key was ``b''``. Both peers did the
        same thing, so round-trip tests passed, and every session ended up
        deriving the same message keys from ``HMAC(b"", ...)``.
        """
        ratchet = create_session_pair()[0]
        new_root, new_chain = ratchet._kdf_rk(b'\x01' * 32, b'\x02' * 32)
        assert len(new_root) == 32
        assert len(new_chain) == 32
        assert new_chain != b''

    def test_chain_keys_are_never_empty_after_a_ratchet(self):
        alice, bob = create_session_pair()
        assert len(alice.state.send_chain_key) == 32

        first, first_hdr = alice.encrypt(b'hello')
        assert bob.decrypt(first, first_hdr) == b'hello'

        # Bob ratcheted on receive, so both of his chains must be full length.
        assert len(bob.state.recv_chain_key) == 32
        assert len(bob.state.send_chain_key) == 32

    def test_distinct_sessions_do_not_share_chain_keys(self):
        """Guards the empty-chain-key failure from the confidentiality side."""
        first_alice, _ = create_session_pair()
        second_alice, _ = create_session_pair()
        assert first_alice.state.send_chain_key != second_alice.state.send_chain_key

        # With an empty chain key these two message keys would be identical.
        first_mk = first_alice._kdf_ck(first_alice.state.send_chain_key)[1]
        second_mk = second_alice._kdf_ck(second_alice.state.send_chain_key)[1]
        assert first_mk != second_mk


class TestStateRollback:
    """
    A message that fails authentication must leave no trace in the state.

    Without rollback, one tampered packet consumes a chain key and bumps the
    counters, which desynchronises the session for good; a forged header with an
    attacker-chosen DH key additionally ratchets the root key before anything is
    authenticated.
    """

    def test_tampered_packet_does_not_desynchronise(self):
        alice, bob = create_session_pair()

        first, first_hdr = alice.encrypt(b"first")
        assert bob.decrypt(first, first_hdr) == b"first"
        snapshot_count = bob.state.recv_msg_count

        second, second_hdr = alice.encrypt(b"second")
        tampered = bytearray(second)
        tampered[40] ^= 0xFF
        with pytest.raises(DecryptionError):
            bob.decrypt(bytes(tampered), second_hdr)

        # State is exactly as it was before the failed attempt.
        assert bob.state.recv_msg_count == snapshot_count
        assert len(bob.state.skipped_keys) == 0

        # And the session keeps working.
        third, third_hdr = alice.encrypt(b"third")
        assert bob.decrypt(third, third_hdr) == b"third"

    def test_forged_dh_header_does_not_destroy_the_session(self):
        import nacl.bindings
        import nacl.utils

        alice, bob = create_session_pair()
        first, first_hdr = alice.encrypt(b"first")
        assert bob.decrypt(first, first_hdr) == b"first"
        root_before = bob.state.root_key
        remote_before = bob.state.dh_remote_pub

        # An unauthenticated header naming a key we have never seen. The old
        # implementation ratcheted the root key with it before any AEAD check.
        nonce = nacl.utils.random(24)
        fake_ct = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            b'forged', b'', nonce, nacl.utils.random(32)
        )
        forged_header = {'dh': nacl.utils.random(32).hex(), 'pn': 0, 'n': 0}
        with pytest.raises(DecryptionError):
            bob.decrypt(nonce + fake_ct, forged_header)

        assert bob.state.root_key == root_before
        assert bob.state.dh_remote_pub == remote_before

        # The legitimate conversation continues unaffected.
        second, second_hdr = alice.encrypt(b"second")
        assert bob.decrypt(second, second_hdr) == b"second"

    def test_replay_does_not_corrupt_state(self):
        alice, bob = create_session_pair()
        enc, hdr = alice.encrypt(b"only once")
        assert bob.decrypt(enc, hdr) == b"only once"
        root_after = bob.state.root_key

        with pytest.raises(DuplicateMessageError):
            bob.decrypt(enc, hdr)
        assert bob.state.root_key == root_after

        following, following_hdr = alice.encrypt(b"next")
        assert bob.decrypt(following, following_hdr) == b"next"


class TestHeaderValidation:
    def _session(self):
        alice, bob = create_session_pair()
        enc, hdr = alice.encrypt(b"hello")
        return bob, enc, hdr

    def test_missing_field_rejected(self):
        bob, enc, hdr = self._session()
        for field in ('dh', 'pn', 'n'):
            broken = dict(hdr)
            broken.pop(field)
            with pytest.raises(InvalidSessionStateError):
                bob.decrypt(enc, broken)

    def test_bad_dh_length_rejected(self):
        bob, enc, hdr = self._session()
        with pytest.raises(InvalidSessionStateError):
            bob.decrypt(enc, dict(hdr, dh='abcd'))

    def test_negative_message_number_rejected(self):
        bob, enc, hdr = self._session()
        with pytest.raises(InvalidSessionStateError):
            bob.decrypt(enc, dict(hdr, n=-1))

    def test_non_integer_message_number_rejected(self):
        bob, enc, hdr = self._session()
        with pytest.raises(InvalidSessionStateError):
            bob.decrypt(enc, dict(hdr, n='0'))

    def test_boolean_is_not_an_integer(self):
        bob, enc, hdr = self._session()
        with pytest.raises(InvalidSessionStateError):
            bob.decrypt(enc, dict(hdr, n=True))

    def test_oversized_message_number_rejected(self):
        bob, enc, hdr = self._session()
        with pytest.raises(InvalidSessionStateError):
            bob.decrypt(enc, dict(hdr, n=2 ** 32))

    def test_short_ciphertext_rejected(self):
        bob, _, hdr = self._session()
        with pytest.raises(DecryptionError):
            bob.decrypt(b'\x00' * 8, hdr)


class TestSkippedKeyBounds:
    def test_out_of_order_delivery_loses_no_keys(self):
        """
        Retained keys must be exactly the ones still needed.

        The previous eviction took ``min()`` over ``(dh_hex, msg_num)`` tuples,
        a lexicographic minimum over hex strings rather than the oldest key, and
        deleted it silently.
        """
        alice, bob = create_session_pair()
        messages = [f"Msg {i}".encode() for i in range(5)]
        encrypted = [alice.encrypt(m) for m in messages]

        for index in (4, 0, 3, 1, 2):
            assert bob.decrypt(*encrypted[index]) == messages[index]
        assert len(bob.state.skipped_keys) == 0

    def test_aggregate_store_is_bounded(self):
        """Exceeding the skip budget raises instead of deriving keys blindly."""
        import nacl.bindings
        import nacl.utils

        alice, bob = create_session_pair()
        enc, hdr = alice.encrypt(b"hello")
        bob.decrypt(enc, hdr)

        nonce = nacl.utils.random(24)
        fake = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            b'x', b'', nonce, nacl.utils.random(32)
        )
        # One past the per-chain budget: refused, and nothing is retained.
        with pytest.raises(TooManySkippedMessagesError):
            bob.decrypt(nonce + fake, {'dh': hdr['dh'], 'pn': 0, 'n': MAX_SKIP + 2})
        assert len(bob.state.skipped_keys) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
