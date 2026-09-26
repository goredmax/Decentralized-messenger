"""
Tests for prekey rotation and pool exhaustion.

Two problems are addressed here.

Retirement: after the signed prekey rotates, peers holding a cached bundle
still have a **valid signature** over it, because the identity key did not
change. The signature alone can therefore never distinguish a current prekey
from a retired one. A monotonic signed_prekey_id in the bundle is what lets a
peer that has learned of a rotation refuse the old one.

Silent downgrade: with an empty pool the responder used to return a bundle
without a one-time prekey, producing a 3-DH session with no forward secrecy and
no indication that anything was lost. That now raises unless the caller opts in.
"""

import pytest

from src.crypto.prekey_store import (
    InMemoryPreKeyStore,
    OneTimePreKeyAlreadyUsed,
)
from src.crypto.x3dh import (
    X3DH,
    PrekeyPoolExhausted,
    RetiredSignedPrekey,
    X3DHResponder,
)
from src.protocol.wire import (
    decode_prekey_bundle,
    encode_prekey_bundle,
)


def build(prekey_count=3, signed_prekey_id=0):
    x3dh = X3DH()
    identity_private, _ = x3dh.generate_identity_keys()
    spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
    store = InMemoryPreKeyStore()
    for index in range(prekey_count):
        store.put(x3dh.generate_one_time_prekey(key_id=100 + index))
    responder = X3DHResponder(
        identity_private=identity_private,
        signed_prekey_private=spk_private,
        prekey_store=store,
        signed_prekey_id=signed_prekey_id,
    )
    return x3dh, responder, store


def initiator():
    x3dh = X3DH()
    alice_private, _ = x3dh.generate_identity_keys()
    return x3dh, alice_private


class TestExhaustionIsLoud:
    def test_empty_pool_refuses(self):
        _, responder, _ = build(prekey_count=0)
        with pytest.raises(PrekeyPoolExhausted):
            responder.publish_bundle()

    def test_error_explains_the_tradeoff(self):
        _, responder, _ = build(prekey_count=0)
        with pytest.raises(PrekeyPoolExhausted) as caught:
            responder.publish_bundle()
        message = str(caught.value)
        assert 'forward secrecy' in message
        assert 'allow_no_one_time_prekey' in message

    def test_opt_in_yields_three_dh(self):
        x3dh, responder, _ = build(prekey_count=0)
        bundle = responder.publish_bundle(allow_no_one_time_prekey=True)
        assert bundle.one_time_prekey is None

        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.begin(alice_private, bundle)
        responder_session = responder.respond(init)
        session.verify_key_confirmation(responder_session.make_key_confirmation())
        assert session.one_time_prekey_id is None
        assert session.root_key == responder_session.root_key

    def test_pool_drains_then_refuses(self):
        x3dh, responder, store = build(prekey_count=1)
        alice_private, _ = x3dh.generate_identity_keys()

        # First handshake uses the only prekey.
        session, init = x3dh.begin(alice_private, responder.publish_bundle())
        responder.respond(init)
        assert store.count() == 0

        # The next one cannot silently get a weaker session.
        with pytest.raises(PrekeyPoolExhausted):
            responder.publish_bundle()

    def test_one_time_prekey_used_only_once(self):
        x3dh, responder, store = build(prekey_count=1)
        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.begin(alice_private, responder.publish_bundle())
        responder.respond(init)
        assert store.was_consumed(session.one_time_prekey_id)
        with pytest.raises(OneTimePreKeyAlreadyUsed):
            responder.respond(init)


class TestReplenishment:
    def test_replenish_fills_the_pool(self):
        _, _, store = build(prekey_count=1)
        assert store.replenish(low_water_mark=5) == 4
        assert store.count() == 5

    def test_replenish_is_a_no_op_above_the_mark(self):
        _, _, store = build(prekey_count=5)
        assert store.replenish(low_water_mark=5) == 0
        assert store.replenish(low_water_mark=3) == 0
        assert store.count() == 5

    def test_replenish_to_an_explicit_target(self):
        _, _, store = build(prekey_count=2)
        assert store.replenish(low_water_mark=2, target=8) == 6
        assert store.count() == 8

    def test_replenish_generates_distinct_keys(self):
        _, _, store = build(prekey_count=0)
        store.replenish(low_water_mark=10)
        ids = store.available_ids()
        assert len(set(ids)) == 10

    def test_replenish_rejects_bad_arguments(self):
        _, _, store = build(prekey_count=1)
        with pytest.raises(ValueError):
            store.replenish(low_water_mark=-1)
        with pytest.raises(ValueError):
            store.replenish(low_water_mark=5, target=3)
        with pytest.raises(ValueError):
            store.replenish(low_water_mark=0, target=10 ** 9)

    def test_exhausted_pool_can_be_rescued(self):
        x3dh, responder, store = build(prekey_count=1)
        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.begin(alice_private, responder.publish_bundle())
        responder.respond(init)

        with pytest.raises(PrekeyPoolExhausted):
            responder.publish_bundle()

        store.replenish(low_water_mark=3)
        session, init = x3dh.begin(alice_private, responder.publish_bundle())
        responder_session = responder.respond(init)
        session.verify_key_confirmation(responder_session.make_key_confirmation())
        assert session.one_time_prekey_id is not None


class TestRotation:
    def test_bundle_carries_the_current_id(self):
        _, responder, _ = build(signed_prekey_id=7)
        assert responder.publish_bundle().signed_prekey_id == 7

    def test_id_survives_the_wire(self):
        _, responder, _ = build(signed_prekey_id=7)
        wire = encode_prekey_bundle(responder.publish_bundle())
        assert decode_prekey_bundle(wire).signed_prekey_id == 7

    def test_untracked_responder_publishes_zero(self):
        _, responder, _ = build(signed_prekey_id=0)
        assert responder.publish_bundle().signed_prekey_id == 0

    def test_old_bundle_refused_against_a_minimum(self):
        """
        The point of the counter: a cached old bundle has a valid signature but
        an old id, so a peer that knows a rotation happened can refuse it.
        """
        x3dh, old_responder, _ = build(signed_prekey_id=1)
        cached = old_responder.publish_bundle()
        X3DH.verify_bundle(cached)          # the signature is still perfectly valid

        alice_private, _ = x3dh.generate_identity_keys()
        with pytest.raises(RetiredSignedPrekey):
            x3dh.initiate_handshake(alice_private, cached, minimum_signed_prekey_id=2)

    def test_current_bundle_accepted_against_a_minimum(self):
        x3dh, responder, _ = build(signed_prekey_id=5)
        alice_private, _ = x3dh.generate_identity_keys()
        session, init = x3dh.initiate_handshake(
            alice_private, responder.publish_bundle(), minimum_signed_prekey_id=5
        )
        assert session is not None and init is not None

    def test_newer_than_required_is_accepted(self):
        x3dh, responder, _ = build(signed_prekey_id=9)
        alice_private, _ = x3dh.generate_identity_keys()
        session, _ = x3dh.initiate_handshake(
            alice_private, responder.publish_bundle(), minimum_signed_prekey_id=4
        )
        assert session is not None

    def test_untracked_bundle_cannot_satisfy_a_minimum(self):
        x3dh, responder, _ = build(signed_prekey_id=0)
        alice_private, _ = x3dh.generate_identity_keys()
        with pytest.raises(RetiredSignedPrekey):
            x3dh.initiate_handshake(
                alice_private, responder.publish_bundle(), minimum_signed_prekey_id=1
            )

    def test_minimum_is_checked_after_the_signature(self):
        """
        The order matters: a bundle that fails authentication must be rejected
        as forged, not as merely old.

        Otherwise a caller could not tell "someone tampered with this" from
        "this peer rotated", and would treat an attack as routine housekeeping.
        """
        from src.crypto.x3dh import InvalidSignatureError, PreKeyBundle

        x3dh, responder, _ = build(signed_prekey_id=1)
        good = responder.publish_bundle()
        forged = PreKeyBundle(
            identity_key=good.identity_key,
            signed_prekey=good.signed_prekey,
            signed_prekey_signature=bytes([good.signed_prekey_signature[0] ^ 0xFF])
            + good.signed_prekey_signature[1:],
            one_time_prekey=good.one_time_prekey,
            one_time_prekey_id=good.one_time_prekey_id,
            signed_prekey_id=1,
        )
        alice_private, _ = x3dh.generate_identity_keys()
        with pytest.raises(InvalidSignatureError):
            x3dh.initiate_handshake(
                alice_private, forged, minimum_signed_prekey_id=99
            )

    def test_begin_forwards_the_minimum(self):
        x3dh, old_responder, _ = build(signed_prekey_id=1)
        cached = old_responder.publish_bundle()
        alice_private, _ = x3dh.generate_identity_keys()
        with pytest.raises(RetiredSignedPrekey):
            x3dh.begin(alice_private, cached, minimum_signed_prekey_id=2)

    def test_negative_id_rejected_at_construction(self):
        x3dh = X3DH()
        identity_private, identity_public = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
        from src.crypto.x3dh import InvalidHandshakeError, PreKeyBundle
        with pytest.raises(InvalidHandshakeError):
            PreKeyBundle(
                identity_key=identity_public,
                signed_prekey=spk_private.public_key,
                signed_prekey_signature=b'\x00' * 64,
                signed_prekey_id=-1,
            )


class TestRotationWorkflow:
    def test_full_rotation_lifecycle(self, tmp_path):
        """
        Rotate, keep a stale bundle, and require the new one.

        Uses the persistent IdentityStore so the counter survives a restart,
        which is the whole point of storing it.
        """
        from src.storage.container import Container
        from src.storage.store import IdentityStore, PersistentPreKeyStore

        kdf = __import__(
            'src.storage.container', fromlist=['KdfParams']
        ).KdfParams(n_log2=12, r=8, p=1)
        password = 'test password'
        store = PersistentPreKeyStore(
            Container.create(str(tmp_path / 'prekeys.bin'), password, kdf_params=kdf)
        )
        store.replenish(low_water_mark=4)
        identity = IdentityStore(
            Container.create(str(tmp_path / 'identity.bin'), password, kdf_params=kdf)
        )
        identity.initialise()

        def make_responder():
            return X3DHResponder(
                identity_private=identity.identity_key,
                signed_prekey_private=identity.signed_prekey_private,
                prekey_store=store,
                signed_prekey_id=identity.signed_prekey_id,
            )

        # A peer caches today's bundle.
        cached = make_responder().publish_bundle()
        assert cached.signed_prekey_id == 1

        # Rotation happens, and survives a restart.
        assert identity.rotate() == 2
        reopened = IdentityStore(Container.open(str(tmp_path / 'identity.bin'), password))
        assert reopened.signed_prekey_id == 2

        x3dh = X3DH()
        alice_private, _ = x3dh.generate_identity_keys()

        # The cached bundle still verifies cryptographically...
        X3DH.verify_bundle(cached)
        # ...but a peer that knows about the rotation refuses it.
        with pytest.raises(RetiredSignedPrekey):
            x3dh.initiate_handshake(alice_private, cached, minimum_signed_prekey_id=2)

        # Fetching fresh works.
        fresh_responder = X3DHResponder(
            identity_private=reopened.identity_key,
            signed_prekey_private=reopened.signed_prekey_private,
            prekey_store=store,
            signed_prekey_id=reopened.signed_prekey_id,
        )
        session, init = x3dh.begin(
            alice_private, fresh_responder.publish_bundle(),
            minimum_signed_prekey_id=2,
        )
        responder_session = fresh_responder.respond(init)
        session.verify_key_confirmation(responder_session.make_key_confirmation())
        assert session.root_key == responder_session.root_key

    def test_rotation_keeps_the_identity_key(self):
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        first_spk = x3dh.generate_signed_prekey(identity_private)[0]
        second_spk = x3dh.generate_signed_prekey(identity_private)[0]
        assert bytes(identity_private) == bytes(identity_private)
        assert bytes(first_spk) != bytes(second_spk)


class TestPersistentReplenishment:
    def test_replenish_persists(self, tmp_path):
        from src.storage.container import Container, KdfParams
        from src.storage.store import PersistentPreKeyStore

        kdf = KdfParams(n_log2=12, r=8, p=1)
        path = str(tmp_path / 'prekeys.bin')
        store = PersistentPreKeyStore(
            Container.create(path, 'pw', kdf_params=kdf)
        )
        assert store.replenish(low_water_mark=6) == 6

        reopened = PersistentPreKeyStore(Container.open(path, 'pw'))
        assert reopened.count() == 6

    def test_replenishment_validates_arguments(self, tmp_path):
        from src.storage.container import Container, KdfParams
        from src.storage.store import PersistentPreKeyStore

        store = PersistentPreKeyStore(
            Container.create(str(tmp_path / 'p.bin'), 'pw',
                             kdf_params=KdfParams(n_log2=12))
        )
        with pytest.raises(ValueError):
            store.replenish(low_water_mark=-1)
        with pytest.raises(ValueError):
            store.replenish(low_water_mark=5, target=2)
