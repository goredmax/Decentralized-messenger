"""
Tests for the X3DH key agreement: agreement itself, the KDF, and key handling.

Authentication regressions live in ``test_x3dh_mitm.py``.
"""

import pytest
from nacl.public import PrivateKey

from src.crypto.prekey_store import InMemoryPreKeyStore
from src.crypto.x3dh import (
    DEFAULT_CONTEXT,
    X3DH,
    X25519_F,
    X3DHResponder,
    InvalidKeyError,
    PrekeyPoolExhausted,
    PreKeyBundle,
)


def make_responder(prekey_count=3):
    x3dh = X3DH()
    identity_private, identity_public = x3dh.generate_identity_keys()
    spk_private, spk_public, signature = x3dh.generate_signed_prekey(identity_private)
    store = InMemoryPreKeyStore()
    for index in range(prekey_count):
        store.put(x3dh.generate_one_time_prekey(key_id=2000 + index))
    responder = X3DHResponder(
        identity_private=identity_private,
        signed_prekey_private=spk_private,
        prekey_store=store,
    )
    return x3dh, responder, store, identity_public


class TestAgreement:
    def test_four_dh_handshake_agrees(self):
        x3dh, responder, _, _ = make_responder()
        alice_private, _ = x3dh.generate_identity_keys()

        state, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
        responder_state = responder.handle_init(init)

        assert len(state.dh_parts) == 4
        assert state.dh_parts == responder_state.dh_parts
        assert state.master_key() == responder_state.master_key()
        assert len(state.master_key()) == 32

    def test_three_dh_handshake_agrees(self):
        """A bundle with no one-time prekey uses the 3-DH variant."""
        x3dh = X3DH()
        alice_private, _ = x3dh.generate_identity_keys()
        bob_private, bob_public = x3dh.generate_identity_keys()
        spk_private, spk_public, signature = x3dh.generate_signed_prekey(bob_private)

        bundle = PreKeyBundle(
            identity_key=bob_public,
            signed_prekey=spk_public,
            signed_prekey_signature=signature,
        )
        state, init = x3dh.initiate_handshake(alice_private, bundle)
        responder_state = x3dh.receive_handshake(
            bob_private, spk_private, None, init
        )

        assert len(state.dh_parts) == 3
        assert state.dh_parts == responder_state.dh_parts
        assert state.master_key() == responder_state.master_key()

    def test_each_handshake_yields_a_unique_secret(self):
        x3dh, responder, _, _ = make_responder(prekey_count=5)
        alice_private, _ = x3dh.generate_identity_keys()

        secrets = set()
        for _ in range(5):
            state, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
            responder.handle_init(init)
            secrets.add(state.master_key())
        assert len(secrets) == 5

    def test_responder_publishes_the_prekey_it_actually_uses(self):
        """The advertised signature must match the private key used for DH."""
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        store = InMemoryPreKeyStore()
        store.put(x3dh.generate_one_time_prekey(key_id=7))
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=store,
        )
        bundle = responder.publish_bundle()
        assert bytes(bundle.signed_prekey) == bytes(spk_private.public_key)
        X3DH.verify_bundle(bundle)

    def test_bundle_without_prekeys_is_published_without_one(self):
        """
        An empty pool refuses by default and only yields a 3-DH bundle on an
        explicit opt-in, so the loss of forward secrecy is never silent.
        """
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=InMemoryPreKeyStore(),
        )
        with pytest.raises(PrekeyPoolExhausted):
            responder.publish_bundle()

        bundle = responder.publish_bundle(allow_no_one_time_prekey=True)
        assert bundle.one_time_prekey is None
        assert bundle.one_time_prekey_id is None
        X3DH.verify_bundle(bundle)

    def test_opted_in_bundle_yields_three_dh_session(self):
        """The downgrade is real, and it really is the 3-DH variant."""
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=InMemoryPreKeyStore(),
        )
        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.begin(
            alice_private, responder.publish_bundle(allow_no_one_time_prekey=True)
        )
        responder_session = responder.respond(init)
        session.verify_key_confirmation(responder_session.make_key_confirmation())
        assert session.one_time_prekey_id is None
        assert session.root_key == responder_session.root_key

    def test_signed_prekey_id_is_published(self):
        _, responder, _, _ = make_responder()
        assert responder.publish_bundle().signed_prekey_id == 0


class TestKeyDerivation:
    def test_master_key_is_deterministic(self):
        parts = [bytes([index]) * 32 for index in range(4)]
        assert X3DH.derive_master_key(parts) == X3DH.derive_master_key(parts)

    def test_context_separates_keys(self):
        parts = [bytes([index]) * 32 for index in range(4)]
        first = X3DH.derive_master_key(parts, context=b'ctx-a')
        second = X3DH.derive_master_key(parts, context=b'ctx-b')
        assert first != second

    def test_three_and_four_dh_are_unambiguous(self):
        """
        The 3-DH and 4-DH cases must not be distinguishable by buffer length.

        A concatenated buffer of ``F || DH1 || DH2 || DH3`` is 128 bytes, the
        same length as ``DH1 || DH2 || DH3 || DH4``. Passing a single buffer is
        therefore rejected outright, and the two valid forms derive different
        keys from the same first three parts.
        """
        first_three = [bytes([index]) * 32 for index in range(3)]
        four = first_three + [b'\x03' * 32]

        three_key = X3DH.derive_master_key(first_three)
        four_key = X3DH.derive_master_key(four)
        assert three_key != four_key

        with pytest.raises(TypeError):
            X3DH.derive_master_key(X25519_F + b''.join(first_three))
        with pytest.raises(TypeError):
            X3DH.derive_master_key(b''.join(four))

    def test_rejects_wrong_part_count(self):
        with pytest.raises(ValueError):
            X3DH.derive_master_key([b'\x00' * 32] * 2)
        with pytest.raises(ValueError):
            X3DH.derive_master_key([b'\x00' * 32] * 5)

    def test_rejects_wrong_part_length(self):
        with pytest.raises(ValueError):
            X3DH.derive_master_key([b'\x00' * 31, b'\x00' * 32, b'\x00' * 32])

    def test_f_prefix_is_applied_once(self):
        """The F prefix must not depend on how the caller passed the parts."""
        parts = [b'\x01' * 32, b'\x02' * 32, b'\x03' * 32]
        from_list = X3DH.derive_master_key(parts)
        from_tuple = X3DH.derive_master_key(tuple(parts))
        assert from_list == from_tuple

    def test_default_context_is_stable(self):
        assert isinstance(DEFAULT_CONTEXT, bytes) and DEFAULT_CONTEXT


class TestKeyHandling:
    def test_low_order_public_key_is_rejected(self):
        """
        A low-order point yields an all-zero shared secret.

        libsodium refuses most of these outright, so drive the check through a
        key that is accepted by the backend but produces a degenerate result.
        """
        x3dh = X3DH()
        private = PrivateKey.generate()
        with pytest.raises(InvalidKeyError):
            x3dh._dh(private, b'\x00' * 32)

    def test_identity_conversion_is_stable(self):
        """The Ed25519 -> X25519 conversion must be deterministic."""
        from src.crypto.key_management import (
            identity_private_x25519,
            identity_public_x25519,
        )
        signing_key, verify_key = X3DH.generate_identity_keys()
        assert bytes(identity_private_x25519(signing_key).public_key) == \
            bytes(identity_public_x25519(verify_key))
