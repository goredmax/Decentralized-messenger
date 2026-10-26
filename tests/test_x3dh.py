"""
Tests for the X3DH key agreement: agreement itself, the KDF, and key handling.

Authentication regressions live in ``test_x3dh_mitm.py``.
"""

import pytest
from nacl.public import PrivateKey

from src.crypto.prekey_store import InMemoryPreKeyStore
from src.crypto.pq import (
    InMemoryPQPreKeyStore,
    PQPreKeyExhausted,
    stocked_pq_store,
)
from src.crypto.x3dh import (
    DEFAULT_CONTEXT,
    LABEL_EK_IK,
    LABEL_EK_OPK,
    LABEL_EK_SPK,
    LABEL_IK_SPK,
    LABEL_KEM,
    X3DH,
    X3DHResponder,
    DhPart,
    InvalidHandshakeError,
    InvalidKeyError,
    PrekeyPoolExhausted,
    PreKeyBundle,
)


def make_responder(prekey_count=3, pq_prekey_count=4):
    """
    A responder with stocked pools.

    ``pq_prekey_count=None`` builds one with no post-quantum pool at all, which
    is the case a caller has to opt into explicitly.
    """
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
        pq_prekey_store=(
            None if pq_prekey_count is None else stocked_pq_store(pq_prekey_count)
        ),
    )
    return x3dh, responder, store, identity_public


class TestAgreement:
    def test_pqxdh_handshake_agrees(self):
        """
        The full case: one-time X25519 prekey *and* an ML-KEM prekey, so five
        contributions, and both sides derive the same root key.
        """
        x3dh, responder, _, _ = make_responder()
        alice_private, _ = x3dh.generate_identity_keys()

        state, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
        responder_state = responder.handle_init(init)

        assert len(state.dh_parts) == 5
        assert [part.label for part in state.dh_parts] == [
            LABEL_IK_SPK,
            LABEL_EK_IK,
            LABEL_EK_SPK,
            LABEL_EK_OPK,
            LABEL_KEM,
        ]
        assert state.dh_parts == responder_state.dh_parts
        assert state.master_key() == responder_state.master_key()
        assert len(state.master_key()) == 32
        assert state.is_post_quantum
        assert responder_state.is_post_quantum

    def test_four_dh_handshake_agrees(self):
        """Without a post-quantum prekey it is the 4-DH variant, and it says so."""
        x3dh, responder, _, _ = make_responder(pq_prekey_count=None)
        alice_private, _ = x3dh.generate_identity_keys()

        state, init = x3dh.initiate_handshake(
            alice_private, responder.publish_bundle(allow_no_pq=True)
        )
        responder_state = responder.handle_init(init)

        assert len(state.dh_parts) == 4
        assert state.dh_parts == responder_state.dh_parts
        assert state.master_key() == responder_state.master_key()
        assert not state.is_post_quantum

    def test_a_post_quantum_prekey_alone_is_four_dh(self):
        """
        An ML-KEM prekey with no X25519 one still has forward secrecy against a
        future X25519 break, which is the case this property exists for.
        """
        x3dh, responder, _, _ = make_responder(prekey_count=0, pq_prekey_count=3)
        alice_private, _ = x3dh.generate_identity_keys()
        bundle = responder.publish_bundle(allow_no_one_time_prekey=True)

        state, init = x3dh.initiate_handshake(alice_private, bundle)
        responder_state = responder.handle_init(init)

        assert [part.label for part in state.dh_parts] == [
            LABEL_IK_SPK,
            LABEL_EK_IK,
            LABEL_EK_SPK,
            LABEL_KEM,
        ]
        assert state.master_key() == responder_state.master_key()
        assert state.is_post_quantum

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
        responder_state = x3dh.receive_handshake(bob_private, spk_private, None, init)

        assert len(state.dh_parts) == 3
        assert state.dh_parts == responder_state.dh_parts
        assert state.master_key() == responder_state.master_key()

    def test_each_handshake_yields_a_unique_secret(self):
        x3dh, responder, _, _ = make_responder(prekey_count=5, pq_prekey_count=5)
        alice_private, _ = x3dh.generate_identity_keys()

        secrets = set()
        for _ in range(5):
            state, init = x3dh.initiate_handshake(
                alice_private, responder.publish_bundle()
            )
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
            pq_prekey_store=stocked_pq_store(2),
        )
        bundle = responder.publish_bundle()
        assert bytes(bundle.signed_prekey) == bytes(spk_private.public_key)
        X3DH.verify_bundle(bundle)

    def test_a_responder_without_a_post_quantum_pool_refuses_by_default(self):
        """
        The loud failure, and the reason it is loud.

        A bundle with no ML-KEM key produces a session whose forward secrecy
        ends the day X25519 is broken, and the peer cannot tell. So the default
        is to refuse and the downgrade has to be asked for by name, exactly as
        the 3-DH downgrade is.
        """
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        store = InMemoryPreKeyStore()
        store.put(x3dh.generate_one_time_prekey(key_id=1))
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=store,
        )
        assert not responder.is_post_quantum
        with pytest.raises(PQPreKeyExhausted):
            responder.publish_bundle()

        bundle = responder.publish_bundle(allow_no_pq=True)
        assert bundle.pq_prekey is None
        assert bundle.pq_prekey_id is None
        X3DH.verify_bundle(bundle)

    def test_an_empty_post_quantum_pool_refuses_like_the_x25519_one(self):
        """
        The X25519 pool has to be deep enough that the *post-quantum* pool is
        what runs out, otherwise this would be testing the other error.
        """
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        store = InMemoryPreKeyStore()
        store.put(x3dh.generate_one_time_prekey(key_id=1))
        store.put(x3dh.generate_one_time_prekey(key_id=2))
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=store,
            pq_prekey_store=stocked_pq_store(1),
        )
        responder.publish_bundle()
        with pytest.raises(PQPreKeyExhausted):
            responder.publish_bundle()

    def test_a_failed_publish_does_not_consume_the_x25519_pool(self):
        """
        The regression this ordering exists for.

        Reserving the X25519 prekey and then failing on the post-quantum pool
        would leave that prekey held out of circulation for a bundle that was
        never published. A client refilling its pool on a low-water mark would
        then lose one prekey per failed attempt until it had none -- and anyone
        who could keep the post-quantum pool empty could trigger that.
        """
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        store = InMemoryPreKeyStore()
        store.put(x3dh.generate_one_time_prekey(key_id=1))
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=store,
        )
        for _ in range(5):
            with pytest.raises(PQPreKeyExhausted):
                responder.publish_bundle()
        assert store.count() == 1
        assert store.reserved_ids() == []
        # And the prekey is still there to be used once the caller accepts the
        # downgrade.
        assert responder.publish_bundle(allow_no_pq=True).one_time_prekey_id == 1

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
            pq_prekey_store=stocked_pq_store(1),
        )
        with pytest.raises(PrekeyPoolExhausted):
            responder.publish_bundle()

        bundle = responder.publish_bundle(allow_no_one_time_prekey=True)
        assert bundle.one_time_prekey is None
        assert bundle.one_time_prekey_id is None
        X3DH.verify_bundle(bundle)

    def test_opted_in_bundle_yields_three_dh_session(self):
        """
        The downgrade is real, and it really is the 3-DH variant -- with both
        prekey pools emptied, so three contributions and nothing else.
        """
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        responder = X3DHResponder(
            identity_private=identity_private,
            signed_prekey_private=spk_private,
            prekey_store=InMemoryPreKeyStore(),
            pq_prekey_store=InMemoryPQPreKeyStore(),
        )
        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.begin(
            alice_private,
            responder.publish_bundle(allow_no_one_time_prekey=True, allow_no_pq=True),
        )
        responder_session = responder.respond(init)
        session.verify_key_confirmation(responder_session.make_key_confirmation())
        assert session.one_time_prekey_id is None
        assert session.pq_prekey_id is None
        assert not session.is_post_quantum
        assert session.root_key == responder_session.root_key

    def test_signed_prekey_id_is_published(self):
        _, responder, _, _ = make_responder()
        assert responder.publish_bundle().signed_prekey_id == 0


class TestKeyDerivation:
    """
    The KDF, and the property that makes the new labelling worth having.

    Before PQXDH every contribution was a bare 32-byte value and the count was
    restricted to 3 or 4. With an ML-KEM secret in the mix that restriction stops
    being enough: "three X25519 outputs plus a one-time prekey" and "three
    X25519 outputs plus an ML-KEM secret" are both four 32-byte values, and a KDF
    handed a buffer cannot tell them apart. The labels exist for that, and the
    tests below are the reason they are not decoration.
    """

    @staticmethod
    def parts(count: int, include_kem: bool = False):
        """A plausible set of contributions, labelled the way a real one is."""
        labels = [LABEL_IK_SPK, LABEL_EK_IK, LABEL_EK_SPK, LABEL_EK_OPK][:count]
        if include_kem:
            labels = labels[:3] + [LABEL_KEM]
        return [
            DhPart(label, bytes([index + 1]) * 32) for index, label in enumerate(labels)
        ]

    def test_master_key_is_deterministic(self):
        parts = self.parts(4)
        assert X3DH.derive_master_key(parts) == X3DH.derive_master_key(parts)

    def test_context_separates_keys(self):
        parts = self.parts(4)
        first = X3DH.derive_master_key(parts, context=b"ctx-a")
        second = X3DH.derive_master_key(parts, context=b"ctx-b")
        assert first != second

    def test_a_buffer_is_rejected(self):
        """
        A concatenated buffer is ambiguous and is refused outright.

        ``F || DH1 || DH2 || DH3`` and ``DH1 || DH2 || DH3 || DH4`` are both 128
        bytes, so length-sniffing a buffer tells the 3-DH and 4-DH cases apart
        wrongly and derives different keys on the two sides.
        """
        parts = self.parts(4)
        with pytest.raises(TypeError):
            X3DH.derive_master_key(b"".join(part.value for part in parts))

    def test_bare_outputs_are_rejected(self):
        """
        The labels are not optional.

        A bare 32-byte output says nothing about which key produced it, so
        accepting one would let an ML-KEM contribution be swapped for an X25519
        one without the KDF noticing. This is the guard that stops a caller from
        quietly reintroducing the ambiguity.
        """
        with pytest.raises(TypeError):
            X3DH.derive_master_key([b"\x01" * 32] * 4)

    def test_an_ml_kem_contribution_is_distinguishable_from_a_prekey_one(self):
        """
        The property the labels buy, stated as the failure it prevents.

        Same three X25519 contributions, same lengths, same total buffer size:
        only the fourth contribution differs, and it differs in *kind* rather
        than in value. Without the label these two handshakes would derive the
        same root key, and a responder that dropped the post-quantum contribution
        would be indistinguishable from one that kept it.
        """
        with_prekey = X3DH.derive_master_key(self.parts(4))
        with_kem = X3DH.derive_master_key(self.parts(3, include_kem=True))
        assert with_prekey != with_kem

        # And the buffer lengths really are equal, which is the whole problem.
        prekey_part = self.parts(4)[3]
        kem_part = self.parts(3, include_kem=True)[3]
        assert len(prekey_part.value) == len(kem_part.value)

    def test_a_different_label_for_the_same_value_is_a_different_key(self):
        first = X3DH.derive_master_key(
            [
                DhPart(LABEL_EK_OPK, b"\x07" * 32),
            ]
        )
        second = X3DH.derive_master_key(
            [
                DhPart(LABEL_KEM, b"\x07" * 32),
            ]
        )
        assert first != second

    def test_duplicate_labels_are_rejected(self):
        """
        Two contributions with one label means the caller assembled the
        handshake wrong, and mixing it in would produce a key neither this
        implementation nor the specification has an opinion about.
        """
        with pytest.raises(ValueError):
            X3DH.derive_master_key(
                [
                    DhPart(LABEL_EK_SPK, b"\x01" * 32),
                    DhPart(LABEL_EK_SPK, b"\x02" * 32),
                ]
            )

    def test_rejects_an_empty_part_list(self):
        with pytest.raises(ValueError):
            X3DH.derive_master_key([])

    def test_rejects_a_part_of_the_wrong_length(self):
        """
        The length is checked where the part is built, not only where it is
        used. A ``DhPart`` that somehow carried 31 bytes would be caught again by
        ``derive_master_key``, and both checks are wanted: one stops it being
        created, the other stops it being used.
        """
        with pytest.raises(InvalidHandshakeError):
            DhPart(LABEL_IK_SPK, b"\x00" * 31)

        forged = DhPart(LABEL_IK_SPK, b"\x00" * 32)
        object.__setattr__(forged, "value", b"\x00" * 31)
        with pytest.raises(ValueError):
            X3DH.derive_master_key([forged])

    def test_f_prefix_is_applied_once(self):
        """The F prefix must not depend on how the caller passed the parts."""
        parts = self.parts(3)
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
            x3dh._dh(private, b"\x00" * 32)

    def test_identity_conversion_is_stable(self):
        """The Ed25519 -> X25519 conversion must be deterministic."""
        from src.crypto.key_management import (
            identity_private_x25519,
            identity_public_x25519,
        )

        signing_key, verify_key = X3DH.generate_identity_keys()
        assert bytes(identity_private_x25519(signing_key).public_key) == bytes(
            identity_public_x25519(verify_key)
        )
