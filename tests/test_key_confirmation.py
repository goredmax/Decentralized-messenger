"""
Tests for X3DH key confirmation.

Without confirmation the initiator cannot distinguish a real peer from a
responder that derived a different secret and said nothing: the handshake looks
successful and the failure surfaces later as undecryptable messages.
Confirmation makes that state unreachable, by refusing to hand out the root key
until the responder's MAC verifies.
"""

import pytest
from nacl.public import PrivateKey

from src.crypto.prekey_store import (
    InMemoryPreKeyStore,
    OneTimePreKeyAlreadyUsed,
)
from src.crypto.x3dh import (
    X3DH,
    InvalidHandshakeError,
    KeyConfirmation,
    KeyConfirmationFailed,
    KeyConfirmationRequired,
    X3DHResponder,
    confirmation_mac,
    confirmation_mac_key,
)


def build(prekey=True, associated_data=b'', prekey_id=1):
    """Build a responder and an unconfirmed initiator session."""
    x3dh = X3DH()
    bob_private, _ = x3dh.generate_identity_keys()
    spk_private, _, _ = x3dh.generate_signed_prekey(bob_private)
    store = InMemoryPreKeyStore()
    if prekey:
        store.put(x3dh.generate_one_time_prekey(key_id=prekey_id))
    responder = X3DHResponder(
        identity_private=bob_private,
        signed_prekey_private=spk_private,
        prekey_store=store,
    )
    alice_private, _ = x3dh.generate_identity_keys()
    session, init = x3dh.begin(
        alice_private, responder.publish_bundle(), associated_data
    )
    return x3dh, session, init, responder


class TestConfirmationRequired:
    def test_initiator_starts_unconfirmed(self):
        _, session, _, _ = build()
        assert not session.is_confirmed

    def test_root_key_locked_before_confirmation(self):
        _, session, _, _ = build()
        with pytest.raises(KeyConfirmationRequired):
            session.root_key

    def test_responder_session_is_confirmed(self):
        """
        The responder has nothing to verify: it authenticated the initiator and
        computed the secret itself. It learns the initiator received the
        confirmation when the first ratchet message decrypts.
        """
        _, _, init, responder = build()
        assert responder.respond(init).is_confirmed

    def test_full_flow_unlocks_and_keys_match(self):
        _, session, init, responder = build()

        with pytest.raises(KeyConfirmationRequired):
            session.root_key

        responder_session = responder.respond(init)
        assert responder_session.is_confirmed

        session.verify_key_confirmation(responder_session.make_key_confirmation())

        assert session.is_confirmed
        assert session.root_key == responder_session.root_key

    def test_confirmation_survives_the_wire(self):
        """The gated flow must work over encoded messages, not just in-process."""
        from src.protocol.wire import (
            decode_key_confirmation,
            encode_handshake_init,
            encode_key_confirmation,
        )

        _, session, init, responder = build()
        init_wire = encode_handshake_init(init)
        responder_session = responder.respond(init)
        wire = encode_key_confirmation(responder_session.make_key_confirmation())

        session.verify_key_confirmation(decode_key_confirmation(wire))
        assert session.is_confirmed
        del init_wire


class TestConfirmationRejection:
    def _pair(self, associated_data=b''):
        _, session, init, responder = build(associated_data=associated_data)
        return session, responder.respond(init, associated_data)

    def test_forged_mac_rejected(self):
        session, _ = self._pair()
        forged = KeyConfirmation(
            ephemeral_key=PrivateKey.generate().public_key,
            mac=b'\x00' * 32,
        )
        with pytest.raises(KeyConfirmationFailed):
            session.verify_key_confirmation(forged)

    def test_tampered_mac_rejected(self):
        session, responder_session = self._pair()
        confirmation = responder_session.make_key_confirmation()
        tampered = KeyConfirmation(
            ephemeral_key=confirmation.ephemeral_key,
            mac=bytes([confirmation.mac[0] ^ 0xFF]) + confirmation.mac[1:],
        )
        with pytest.raises(KeyConfirmationFailed):
            session.verify_key_confirmation(tampered)

    def test_swapped_ephemeral_key_rejected(self):
        """
        The MAC covers the responder's ephemeral key, so a captured confirmation
        cannot be replayed with a different key swapped in.
        """
        session, responder_session = self._pair()
        confirmation = responder_session.make_key_confirmation()
        swapped = KeyConfirmation(
            ephemeral_key=PrivateKey.generate().public_key,
            mac=confirmation.mac,
        )
        with pytest.raises(KeyConfirmationFailed):
            session.verify_key_confirmation(swapped)

    def test_associated_data_mismatch_rejected(self):
        session, responder_session = self._pair(associated_data=b'ctx-a')
        confirmation = responder_session.make_key_confirmation()
        # Same transcript, but the initiator believes the context differs.
        session._associated_data = b'ctx-b'
        with pytest.raises(KeyConfirmationFailed):
            session.verify_key_confirmation(confirmation)

    def test_confirmation_from_another_session_rejected(self):
        first, _ = self._pair()
        _, second_responder = self._pair()
        with pytest.raises(KeyConfirmationFailed):
            first.verify_key_confirmation(second_responder.make_key_confirmation())

    def test_wrong_type_rejected(self):
        session, _ = self._pair()
        with pytest.raises(InvalidHandshakeError):
            session.verify_key_confirmation(b'not-a-confirmation')

    def test_short_mac_rejected_at_construction(self):
        with pytest.raises(KeyConfirmationFailed):
            KeyConfirmation(
                ephemeral_key=PrivateKey.generate().public_key,
                mac=b'\x00' * 16,
            )

    def test_short_ephemeral_key_rejected_at_construction(self):
        with pytest.raises(InvalidHandshakeError):
            KeyConfirmation(ephemeral_key=b'\x00' * 16, mac=b'\x00' * 32)


class TestDirectionalRules:
    def test_initiator_cannot_make_confirmation(self):
        _, session, _, _ = build()
        with pytest.raises(InvalidHandshakeError):
            session.make_key_confirmation()

    def test_responder_cannot_verify(self):
        _, _, init, responder = build()
        responder_session = responder.respond(init)
        with pytest.raises(InvalidHandshakeError):
            responder_session.verify_key_confirmation(
                responder_session.make_key_confirmation()
            )


class TestMacDerivation:
    def test_mac_key_depends_on_root_key(self):
        assert confirmation_mac_key(b'\x00' * 32) != confirmation_mac_key(b'\x01' * 32)

    def test_mac_key_is_deterministic(self):
        assert confirmation_mac_key(b'\x42' * 32) == confirmation_mac_key(b'\x42' * 32)

    def test_mac_key_requires_32_bytes(self):
        with pytest.raises(InvalidHandshakeError):
            confirmation_mac_key(b'\x00' * 16)

    def test_mac_binds_every_transcript_field(self):
        """
        Each field must change the MAC, or it is not actually bound.

        Without this a confirmation could be reused against a different identity
        key, a different ephemeral key, or different associated data.
        """
        root = b'\x07' * 32
        _, alice_identity = X3DH.generate_identity_keys()
        _, bob_identity = X3DH.generate_identity_keys()
        alice_ephemeral = PrivateKey.generate().public_key
        bob_ephemeral = PrivateKey.generate().public_key

        def mac(**overrides):
            kwargs = dict(
                associated_data=b'ad',
                alice_identity=alice_identity,
                alice_ephemeral=alice_ephemeral,
                bob_identity=bob_identity,
                bob_ephemeral=bob_ephemeral,
            )
            kwargs.update(overrides)
            return confirmation_mac(root, **kwargs)

        baseline = mac()
        assert baseline != mac(associated_data=b'other')
        assert baseline != mac(alice_identity=X3DH.generate_identity_keys()[1])
        assert baseline != mac(alice_ephemeral=PrivateKey.generate().public_key)
        assert baseline != mac(bob_identity=X3DH.generate_identity_keys()[1])
        assert baseline != mac(bob_ephemeral=PrivateKey.generate().public_key)

    def test_mac_depends_on_root_key(self):
        _, alice_identity = X3DH.generate_identity_keys()
        _, bob_identity = X3DH.generate_identity_keys()
        kwargs = dict(
            associated_data=b'',
            alice_identity=alice_identity,
            alice_ephemeral=PrivateKey.generate().public_key,
            bob_identity=bob_identity,
            bob_ephemeral=PrivateKey.generate().public_key,
        )
        assert confirmation_mac(b'\x01' * 32, **kwargs) != \
            confirmation_mac(b'\x02' * 32, **kwargs)


class TestPrekeyStillSingleUse:
    def test_respond_consumes_prekey_once(self):
        _, _, init, responder = build()
        responder.respond(init)
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            responder.respond(init)

    def test_forged_handshake_still_does_not_spend_prekey(self):
        from dataclasses import replace

        _, _, init, responder = build()
        forged = replace(init, ephemeral_key=PrivateKey.generate().public_key)
        with pytest.raises(Exception):
            responder.respond(forged)
        # The genuine handshake must still be able to use the prekey.
        assert responder.respond(init).is_confirmed
