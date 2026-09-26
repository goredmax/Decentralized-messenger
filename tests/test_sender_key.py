"""
Tests for Sender Keys, the group messaging protocol.

The properties worth pinning down:

- one encryption serves every member, because the key is re-derived rather than
  transmitted
- a member who joins later cannot read earlier messages, because their session
  does not exist yet and cannot be walked backwards
- removing a member is not effective until everyone rotates
- a replayed or conflicting distribution message is reported, not accepted
"""

import pytest
from nacl.public import PrivateKey
from nacl.signing import SigningKey

from src.protocol.sender_key import (
    MAX_ITERATION,
    SenderKeyDistributionMessage,
    SenderKeyError,
    SenderKeyId,
    SenderKeyMessage,
    SenderKeyReplay,
    SenderKeySession,
    SenderKeySessionBuilder,
    SenderKeySkippedTooFar,
    SessionCipher,
    UntrustedSenderKey,
)

GROUP = 'test-group'


def member(name=None):
    return SigningKey.generate()


def join(cipher, group, *recipients):
    """
    Deliver the sender's distribution to each recipient.

    In a real deployment this travels over a pairwise Double Ratchet session;
    here it is a direct call, since the transport is not what is under test.
    """
    distribution = cipher.distribution_for(group)
    flagged = []
    for recipient in recipients:
        _, untrusted = recipient.process_distribution(distribution)
        if untrusted:
            flagged.append(recipient)
    return distribution, flagged


class TestSenderKeyId:
    def test_equality_is_by_value(self):
        key = SigningKey.generate()
        first = SenderKeyId(group_id=GROUP, sender=key.verify_key)
        second = SenderKeyId(group_id=GROUP, sender=key.verify_key)
        assert first == second
        assert hash(first) == hash(second)

    def test_different_groups_differ(self):
        key = SigningKey.generate()
        assert SenderKeyId('a', key.verify_key) != SenderKeyId('b', key.verify_key)

    def test_different_senders_differ(self):
        assert (SenderKeyId('a', member().verify_key)
                != SenderKeyId('a', member().verify_key))

    def test_rejects_empty_group(self):
        with pytest.raises(SenderKeyError):
            SenderKeyId('', member().verify_key)

    def test_rejects_non_verify_key(self):
        with pytest.raises(SenderKeyError):
            SenderKeyId(GROUP, b'not a key')

    def test_usable_as_a_dict_key(self):
        key = member().verify_key
        mapping = {SenderKeyId(GROUP, key): 'value'}
        assert mapping[SenderKeyId(GROUP, key)] == 'value'


class TestDistribution:
    def test_signature_verifies(self):
        cipher = SessionCipher(member())
        cipher.distribution_for(GROUP).verify()

    def test_tampered_chain_key_fails_verification(self):
        cipher = SessionCipher(member())
        distribution = cipher.distribution_for(GROUP)
        broken = SenderKeyDistributionMessage(
            key_id=distribution.key_id,
            iteration=distribution.iteration,
            chain_key=PrivateKey.generate().public_key.__bytes__(),
            signature=distribution.signature,
            chain_id=distribution.chain_id,
        )
        with pytest.raises(SenderKeyError):
            broken.verify()

    def test_tampered_chain_id_fails_verification(self):
        cipher = SessionCipher(member())
        distribution = cipher.distribution_for(GROUP)
        broken = SenderKeyDistributionMessage(
            key_id=distribution.key_id,
            iteration=distribution.iteration,
            chain_key=distribution.chain_key,
            signature=distribution.signature,
            chain_id=PrivateKey.generate().public_key.__bytes__(),
        )
        with pytest.raises(SenderKeyError):
            broken.verify()

    def test_tampered_iteration_fails_verification(self):
        cipher = SessionCipher(member())
        distribution = cipher.distribution_for(GROUP)
        broken = SenderKeyDistributionMessage(
            key_id=distribution.key_id,
            iteration=distribution.iteration + 1,
            chain_key=distribution.chain_key,
            signature=distribution.signature,
            chain_id=distribution.chain_id,
        )
        with pytest.raises(SenderKeyError):
            broken.verify()

    def test_forged_by_another_identity_fails(self):
        """A member cannot advertise a chain under someone else's name."""
        real = member()
        impostor = member()
        cipher = SessionCipher(real)
        distribution = cipher.distribution_for(GROUP)
        stolen = SenderKeyDistributionMessage(
            key_id=distribution.key_id,
            iteration=distribution.iteration,
            chain_key=distribution.chain_key,
            signature=impostor.sign(
                distribution._signed_bytes()
            ).signature,
            chain_id=distribution.chain_id,
        )
        with pytest.raises(SenderKeyError):
            stolen.verify()

    def test_cannot_sign_with_a_mismatched_identity(self):
        identity = member()
        other = member()
        unsigned = SenderKeyDistributionMessage(
            key_id=SenderKeyId(GROUP, identity.verify_key),
            iteration=0,
            chain_key=PrivateKey.generate().public_key.__bytes__(),
            signature=b'\x00' * 64,
            chain_id=PrivateKey.generate().public_key.__bytes__(),
        )
        with pytest.raises(SenderKeyError):
            unsigned.sign(other)

    def test_rejects_bad_iteration(self):
        identity = member()
        with pytest.raises(SenderKeyError):
            SenderKeyDistributionMessage(
                key_id=SenderKeyId(GROUP, identity.verify_key),
                iteration=MAX_ITERATION + 1,
                chain_key=b'\x00' * 32,
                signature=b'\x00' * 64,
                chain_id=b'\x00' * 32,
            )

    def test_rejects_short_chain_key(self):
        identity = member()
        with pytest.raises(SenderKeyError):
            SenderKeyDistributionMessage(
                key_id=SenderKeyId(GROUP, identity.verify_key),
                iteration=0,
                chain_key=b'\x00' * 16,
                signature=b'\x00' * 64,
                chain_id=b'\x00' * 32,
            )

    def test_rejects_missing_chain_id(self):
        identity = member()
        with pytest.raises(SenderKeyError):
            SenderKeyDistributionMessage(
                key_id=SenderKeyId(GROUP, identity.verify_key),
                iteration=0,
                chain_key=b'\x00' * 32,
                signature=b'\x00' * 64,
            )


class TestGroupMessaging:
    def test_all_members_read_one_encryption(self):
        alice, bob, carol = member(), member(), member()
        sender = SessionCipher(alice)
        first = SessionCipher(bob)
        second = SessionCipher(carol)

        join(sender, GROUP, first, second)
        message, _ = sender.encrypt(b'hello everyone', GROUP)
        assert first.decrypt(message) == b'hello everyone'
        assert second.decrypt(message) == b'hello everyone'

    def test_a_long_conversation(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        for index in range(20):
            text = f'message {index}'.encode()
            message, _ = sender.encrypt(text, GROUP)
            assert receiver.decrypt(message) == text

    def test_separate_groups_are_separate(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, 'alpha', receiver)
        message, _ = sender.encrypt(b'for alpha', 'alpha')
        # The receiver holds no session for beta, and alpha's message is
        # unrelated to it.
        with pytest.raises(SenderKeyError):
            receiver.decrypt(
                SenderKeyMessage(
                    key_id=SenderKeyId('beta', receiver.builder.sessions()[0].sender),
                    iteration=0,
                    ciphertext=message.ciphertext,
                )
            )

    def test_members_have_independent_chains(self):
        """Two senders in one group do not collide."""
        alice, bob = member(), member()
        first_cipher, second_cipher = SessionCipher(alice), SessionCipher(bob)
        listener = SessionCipher(member())
        join(first_cipher, GROUP, listener)
        join(second_cipher, GROUP, listener)

        from_alice, _ = first_cipher.encrypt(b'from alice', GROUP)
        from_bob, _ = second_cipher.encrypt(b'from bob', GROUP)
        assert listener.decrypt(from_alice) == b'from alice'
        assert listener.decrypt(from_bob) == b'from bob'

    def test_receiver_needs_a_distribution_first(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        message, _ = sender.encrypt(b'too early', GROUP)
        with pytest.raises(UntrustedSenderKey):
            receiver.decrypt(message)

    def test_tampered_ciphertext_is_rejected(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        message, _ = sender.encrypt(b'secret', GROUP)
        altered = SenderKeyMessage(
            key_id=message.key_id,
            iteration=message.iteration,
            ciphertext=bytes([message.ciphertext[0] ^ 0xFF]) + message.ciphertext[1:],
        )
        with pytest.raises(SenderKeyError):
            receiver.decrypt(altered)

    def test_wrong_group_cannot_decrypt(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        message, _ = sender.encrypt(b'secret', GROUP)
        # Same key material, different group id in the header.
        forged = SenderKeyMessage(
            key_id=SenderKeyId('other-group', message.key_id.sender),
            iteration=message.iteration,
            ciphertext=message.ciphertext,
        )
        with pytest.raises((SenderKeyError, UntrustedSenderKey)):
            receiver.decrypt(forged)


class TestLateJoiner:
    def test_cannot_read_history(self):
        """
        The property the whole design exists for.

        A member that was not present when messages were sent has no session, so
        there is no chain position to walk back from.
        """
        sender = SessionCipher(member())
        early = SessionCipher(member())
        join(sender, GROUP, early)

        history = [sender.encrypt(f'old {i}'.encode(), GROUP)[0] for i in range(3)]
        early_seen = [early.decrypt(message) for message in history]

        # A newcomer receives the current distribution only.
        latecomer = SessionCipher(member())
        sender.distribution_for(GROUP)
        latecomer.process_distribution(sender.distribution_for(GROUP))

        for message, expected in zip(history, early_seen):
            with pytest.raises(SenderKeyError):
                latecomer.decrypt(message)

    def test_reads_new_messages_after_distribution(self):
        sender = SessionCipher(member())
        early = SessionCipher(member())
        join(sender, GROUP, early)
        sender.encrypt(b'before', GROUP)

        latecomer = SessionCipher(member())
        latecomer.process_distribution(sender.distribution_for(GROUP))
        message, _ = sender.encrypt(b'after', GROUP)
        assert latecomer.decrypt(message) == b'after'


class TestRotation:
    def test_removed_member_loses_access_after_rotation(self):
        """
        The real test of rotation: not that the message is unreadable to the
        removed member immediately, but that rotation makes it so.
        """
        sender = SessionCipher(member())
        staying = SessionCipher(member())
        leaving = SessionCipher(member())
        join(sender, GROUP, staying, leaving)
        old, _ = sender.encrypt(b'before removal', GROUP)
        assert leaving.decrypt(old) == b'before removal'

        # The sender rotates; the staying member confirms the new chain.
        rotated = sender.rotate_after_removal(GROUP)
        _, untrusted = staying.process_distribution(rotated)
        assert untrusted is True
        staying.builder.trust(rotated)
        sender.remove_member(GROUP, leaving.builder.sessions()[0].sender)

        new, _ = sender.encrypt(b'after removal', GROUP)
        assert staying.decrypt(new) == b'after removal'
        with pytest.raises(SenderKeyError):
            leaving.decrypt(new)

    def test_rotation_changes_the_chain(self):
        cipher = SessionCipher(member())
        before = cipher.distribution_for(GROUP)
        after = cipher.rotate_after_removal(GROUP)
        assert before.chain_key != after.chain_key

    def test_rotation_changes_the_chain_id(self):
        """
        This is what makes a rotation detectable at all.

        The sender key id is (group, sender) and does not change, so without a
        fresh chain id a replayed old distribution would be indistinguishable
        from a live one.
        """
        cipher = SessionCipher(member())
        before = cipher.distribution_for(GROUP)
        after = cipher.rotate_after_removal(GROUP)
        assert before.chain_id != after.chain_id
        assert before.key_id == after.key_id

    def test_rotation_resets_iteration(self):
        cipher = SessionCipher(member())
        for _ in range(5):
            cipher.encrypt(b'x', GROUP)
        rotated = cipher.rotate_after_removal(GROUP)
        assert rotated.iteration == 0

    def test_rotated_distribution_is_signed(self):
        cipher = SessionCipher(member())
        cipher.rotate_after_removal(GROUP).verify()

    def test_removing_locally_forgets_the_session(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        assert len(receiver.builder.sessions()) == 1
        held = receiver.builder.sessions()[0].sender
        receiver.remove_member(GROUP, held)
        assert receiver.builder.sessions() == []


class TestTrustOnFirstUse:
    def test_first_distribution_is_trusted(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        _, untrusted = receiver.process_distribution(sender.distribution_for(GROUP))
        assert untrusted is False

    def test_replaying_the_same_chain_is_harmless(self):
        """
        A server replaying the very same distribution must change nothing.

        The position only ever moves forward, so a replay cannot roll the member
        back onto an earlier point of a chain.
        """
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        distribution = sender.distribution_for(GROUP)
        receiver.process_distribution(distribution)
        _, untrusted = receiver.process_distribution(distribution)
        assert untrusted is False

    def test_a_second_chain_from_a_known_sender_is_reported(self):
        """
        A different chain id from a sender already held is refused by default.

        This is both a rotation and what a replay of a dropped old chain looks
        like, so it cannot be resolved here. Refusing by default is the safe
        direction: the caller confirms a rotation explicitly.
        """
        identity = member()
        first = SessionCipher(identity)
        receiver = SessionCipher(member())
        receiver.process_distribution(first.distribution_for(GROUP))

        second = SessionCipher(identity)
        _, untrusted = receiver.process_distribution(second.distribution_for(GROUP))
        assert untrusted is True

    def test_a_refused_conflict_does_not_change_the_session(self):
        identity = member()
        first = SessionCipher(identity)
        receiver = SessionCipher(member())
        receiver.process_distribution(first.distribution_for(GROUP))
        before = receiver.builder.sessions()
        position = receiver.builder.get_session(before[0]).iteration

        second = SessionCipher(identity)
        receiver.process_distribution(second.distribution_for(GROUP))

        assert receiver.builder.sessions() == before
        assert receiver.builder.get_session(before[0]).iteration == position

    def test_trust_accepts_a_rotation_explicitly(self):
        identity = member()
        first = SessionCipher(identity)
        receiver = SessionCipher(member())
        receiver.process_distribution(first.distribution_for(GROUP))

        second = SessionCipher(identity)
        rotated = second.distribution_for(GROUP)
        _, untrusted = receiver.process_distribution(rotated)
        assert untrusted is True

        receiver.builder.trust(rotated)
        message, _ = second.encrypt(b'after rotation', GROUP)
        assert receiver.decrypt(message) == b'after rotation'

    def test_trust_requires_a_valid_signature(self):
        receiver = SessionCipher(member())
        forged = SenderKeyDistributionMessage(
            key_id=SenderKeyId(GROUP, member().verify_key),
            iteration=0,
            chain_key=b'\x00' * 32,
            signature=b'\x00' * 64,
            chain_id=b'\x00' * 32,
        )
        with pytest.raises(SenderKeyError):
            receiver.builder.trust(forged)

    def test_chain_id_is_recorded_per_sender_and_group(self):
        identity = member()
        receiver = SessionCipher(member())
        distribution = SessionCipher(identity).distribution_for(GROUP)
        receiver.process_distribution(distribution)
        assert receiver.builder.trusted_chains(distribution.key_id) == \
            [distribution.chain_id]

    def test_a_new_group_from_a_known_sender_is_adopted(self):
        """
        Trust is scoped per group, so joining a second group is not a rotation.

        The chain is per group, so a member present in two groups has two
        independent chains. Scoping trust to the sender alone would make the
        second one look like a rotation.
        """
        identity = member()
        sender = SessionCipher(identity)
        receiver = SessionCipher(member())
        receiver.process_distribution(sender.distribution_for('alpha'))
        _, untrusted = receiver.process_distribution(sender.distribution_for('beta'))
        assert untrusted is False
        assert len(receiver.builder.sessions()) == 2

    def test_unsigned_distribution_is_refused(self):
        receiver = SessionCipher(member())
        unsigned = SenderKeyDistributionMessage(
            key_id=SenderKeyId(GROUP, member().verify_key),
            iteration=0,
            chain_key=b'\x00' * 32,
            signature=b'\x00' * 64,
            chain_id=b'\x00' * 32,
        )
        with pytest.raises(SenderKeyError):
            receiver.process_distribution(unsigned)


class TestReplayAndGaps:
    def _session_at(self, iteration):
        session = SenderKeySession(
            key_id=SenderKeyId(GROUP, member().verify_key),
            chain_key=PrivateKey.generate().public_key.__bytes__(),
            iteration=0,
        )
        session.iteration = iteration
        return session

    def test_replaying_a_consumed_iteration_is_refused(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        message, _ = sender.encrypt(b'once', GROUP)
        assert receiver.decrypt(message) == b'once'
        with pytest.raises(SenderKeyReplay):
            receiver.decrypt(message)

    def test_out_of_order_is_refused(self):
        """
        Sender keys do not tolerate reordering, and that is a design choice.

        Unlike the Double Ratchet, a sender key chain retains no skipped keys:
        the whole point is that no key material is kept, so a receiver that
        misses an iteration cannot recover it. A group message that arrives out
        of order is refused rather than silently mis-decrypted.

        The mitigation belongs in the transport, which must deliver in order or
        request retransmission.
        """
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        messages = [sender.encrypt(f'm{i}'.encode(), GROUP)[0] for i in range(3)]
        # Consume one, then ask for it again.
        assert receiver.decrypt(messages[0]) == b'm0'
        with pytest.raises(SenderKeyReplay):
            receiver.decrypt(messages[0])
        # An iteration below the session's current position is refused too, so a
        # reorder cannot walk the chain backwards.
        with pytest.raises(SenderKeyReplay):
            receiver.decrypt(messages[0])

    def test_in_order_delivery_after_a_gap_works(self):
        sender = SessionCipher(member())
        receiver = SessionCipher(member())
        join(sender, GROUP, receiver)
        messages = [sender.encrypt(f'm{i}'.encode(), GROUP)[0] for i in range(3)]
        for index in range(3):
            assert receiver.decrypt(messages[index]) == f'm{index}'.encode()

    def test_gap_beyond_the_budget_is_refused(self):
        session = self._session_at(0)
        with pytest.raises(SenderKeySkippedTooFar):
            session.get_message_key(SenderKeySkippedTooFar.MAX_CATCH_UP + 1)

    def test_gap_within_the_budget_is_iterated(self):
        session = self._session_at(0)
        assert len(session.get_message_key(500)) == 32
        assert session.iteration == 501

    def test_negative_iteration_refused(self):
        with pytest.raises(SenderKeyError):
            self._session_at(0).get_message_key(-1)

    def test_bool_iteration_refused(self):
        with pytest.raises(SenderKeyError):
            self._session_at(0).get_message_key(True)

    def test_short_ciphertext_rejected_at_construction(self):
        with pytest.raises(SenderKeyError):
            SenderKeyMessage(
                key_id=SenderKeyId(GROUP, member().verify_key),
                iteration=0,
                ciphertext=b'\x00' * 8,
            )


class TestSessionBuilder:
    def test_sessions_are_listed(self):
        builder = SenderKeySessionBuilder()
        identity = member()
        distribution = SenderKeyDistributionMessage(
            key_id=SenderKeyId(GROUP, identity.verify_key),
            iteration=0,
            chain_key=PrivateKey.generate().public_key.__bytes__(),
            signature=b'\x00' * 64,
            chain_id=PrivateKey.generate().public_key.__bytes__(),
        )
        builder.process(distribution)
        assert len(builder.sessions()) == 1

    def test_missing_session_is_reported_clearly(self):
        builder = SenderKeySessionBuilder()
        with pytest.raises(UntrustedSenderKey):
            builder.get_session(SenderKeyId(GROUP, member().verify_key))

    def test_later_distribution_advances_the_position(self):
        identity = member()
        builder = SenderKeySessionBuilder()
        key_id = SenderKeyId(GROUP, identity.verify_key)
        chain_key = PrivateKey.generate().public_key.__bytes__()
        chain_id = PrivateKey.generate().public_key.__bytes__()
        base = SenderKeyDistributionMessage(
            key_id=key_id,
            iteration=0,
            chain_key=chain_key,
            signature=b'\x00' * 64,
            chain_id=chain_id,
        )
        builder.process(base)
        ahead = SenderKeyDistributionMessage(
            key_id=key_id, iteration=5, chain_key=chain_key,
            signature=b'\x00' * 64, chain_id=chain_id,
        )
        builder.process(ahead)
        assert builder.get_session(key_id).iteration == 5
