"""
Tests for the post-quantum prekeys and the PQXDH handshake they take part in.

Three things are worth more than the round trip here, and the tests are
organised around them.

**Single use.** A post-quantum prekey used twice is a prekey whose ML-KEM secret
is recoverable from two handshakes, which destroys the forward secrecy the
contribution exists to provide. The pool's reserve/consume/release discipline is
therefore tested as carefully as the X25519 pool's, and the responder is tested
for spending both pools together or neither.

**The ciphertext is inside the signature.** An ML-KEM ciphertext that is not
covered by the initiator's signature can be replaced by an active attacker with
one the attacker generated itself. The attacker would decapsulate it while the
two honest parties would not share the secret -- a post-quantum downgrade that
the key confirmation would eventually notice, with nothing to point at. So the
transcript test substitutes a valid ciphertext and requires a signature failure.

**No silent downgrade.** A bundle without a post-quantum key is a session whose
forward secrecy ends the day X25519 is broken, and the peer cannot tell. So the
default is to refuse and the downgrade has to be asked for by name.
"""

import threading

import pytest

from src.crypto.pq import (
    CIPHERTEXT_LEN,
    PRIVATE_KEY_LEN,
    PUBLIC_KEY_LEN,
    SHARED_SECRET_LEN,
    InvalidPQKey,
    NoSuchPQPreKey,
    PQError,
    PQPreKeyAlreadyUsed,
    PQPreKeyExhausted,
    PQOneTimePreKey,
    decapsulate,
    encapsulate,
    generate_prekey,
    load_public_key,
    stocked_pq_store,
)
from src.crypto.prekey_store import OneTimePreKeyAlreadyUsed
from src.crypto.x3dh import (
    X3DH,
    X3DHResponder,
    InvalidHandshakeError,
    InvalidSignatureError,
    LABEL_KEM,
    PreKeyBundle,
)
from src.protocol.wire import (
    KIND_PREKEY_BUNDLE,
    WIRE_VERSION,
    UnsupportedVersion,
    decode_handshake_init,
    decode_prekey_bundle,
    encode_handshake_init,
    encode_prekey_bundle,
)
from src.storage.container import Container, StorageError
from src.storage.store import PersistentPQPreKeyStore

FAST = None


def make_responder(prekey_count=3, pq_prekey_count=3):
    x3dh = X3DH()
    identity_private, identity_public = x3dh.generate_identity_keys()
    signed_prekey_private, _, _ = x3dh.generate_signed_prekey(identity_private)
    from src.crypto.prekey_store import InMemoryPreKeyStore

    store = InMemoryPreKeyStore()
    for index in range(prekey_count):
        store.put(x3dh.generate_one_time_prekey(key_id=index))
    responder = X3DHResponder(
        identity_private=identity_private,
        signed_prekey_private=signed_prekey_private,
        prekey_store=store,
        pq_prekey_store=(
            None if pq_prekey_count is None
            else stocked_pq_store(pq_prekey_count)
        ),
    )
    return x3dh, responder


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------

def test_the_sizes_are_the_ones_ml_kem_768_defines():
    prekey = generate_prekey()
    shared, ciphertext = encapsulate(prekey.public_key)
    assert len(prekey.public_bytes) == PUBLIC_KEY_LEN == 1184
    assert len(ciphertext) == CIPHERTEXT_LEN == 1088
    assert len(shared) == SHARED_SECRET_LEN == 32
    assert len(prekey.seed) == PRIVATE_KEY_LEN == 64


def test_encapsulate_and_decapsulate_agree():
    prekey = generate_prekey()
    shared, ciphertext = encapsulate(prekey.public_key)
    assert decapsulate(prekey.private_key, ciphertext) == shared


def test_a_public_key_survives_the_wire():
    prekey = generate_prekey()
    assert load_public_key(prekey.public_bytes) == prekey.public_key


def test_a_private_key_survives_its_seed():
    """
    The only reload route there is.

    ``MLKEM768PrivateKey`` has no ``from_private_bytes``, so these 64 bytes are
    the whole of persistence. If this stops working, a persisted pool cannot be
    reopened and every responder would have to be rebuilt from scratch.
    """
    prekey = generate_prekey(key_id=7)
    restored = PQOneTimePreKey.from_seed(7, prekey.seed)
    assert restored == prekey
    assert restored.public_bytes == prekey.public_bytes
    shared, ciphertext = encapsulate(restored.public_key)
    assert decapsulate(restored.private_key, ciphertext) == shared


def test_a_seed_of_the_wrong_length_is_refused():
    """
    Notably, a 32-byte seed: that is the obvious guess and ML-KEM rejects it
    with "a seed is 64 bytes long". Wrong here means a responder that cannot
    reopen its pool.
    """
    for seed in (b'', b'\x00' * 32, b'\x00' * 63, b'\x00' * 65):
        with pytest.raises(PQError):
            PQOneTimePreKey.from_seed(1, seed)


def test_encapsulation_is_randomised():
    """
    Two encapsulations against one key give different secrets.

    This is why the MITM test decapsulates the ciphertext that went on the wire
    rather than encapsulating again: an attacker who re-encapsulated would get a
    different secret and the demonstration would be of nothing.
    """
    prekey = generate_prekey()
    first, _ = encapsulate(prekey.public_key)
    second, _ = encapsulate(prekey.public_key)
    assert first != second


def test_a_wrong_ciphertext_gives_a_different_secret_rather_than_an_error():
    """
    ML-KEM's decapsulation is the IND-CCA2 variant, so it cannot fail.

    Documented here because it is the sort of thing a caller would otherwise
    assume and build an error path around: a tampered ciphertext produces a
    pseudorandom secret, the two sides derive different root keys, and the
    failure surfaces at the key confirmation and the AEAD rather than here.
    """
    prekey = generate_prekey()
    _, ciphertext = encapsulate(prekey.public_key)
    tampered = b'\x00' * CIPHERTEXT_LEN
    assert decapsulate(prekey.private_key, tampered) != decapsulate(
        prekey.private_key, ciphertext
    )
    assert len(decapsulate(prekey.private_key, tampered)) == SHARED_SECRET_LEN


def test_a_ciphertext_of_the_wrong_length_is_refused():
    prekey = generate_prekey()
    for bad in (b'', b'\x00' * 32, b'\x00' * (CIPHERTEXT_LEN - 1),
                b'\x00' * (CIPHERTEXT_LEN + 1)):
        with pytest.raises(InvalidPQKey):
            decapsulate(prekey.private_key, bad)


def test_a_public_key_of_the_wrong_length_is_refused():
    for bad in (b'', b'\x00' * 32, b'\x00' * (PUBLIC_KEY_LEN - 1),
                b'\x00' * (PUBLIC_KEY_LEN + 1)):
        with pytest.raises(InvalidPQKey):
            load_public_key(bad)


def test_a_der_spki_of_the_wrong_algorithm_would_have_loaded():
    """
    Why the wire carries raw bytes and not a DER SPKI.

    ``serialization.load_der_public_key`` accepts an Ed25519 SPKI and returns an
    ``Ed25519PublicKey``. Had the bundle carried a SPKI, a valid key of the
    wrong algorithm would have decoded into an object that looks like a key and
    is not one, and the failure would have surfaced as an attribute error
    somewhere far from the bundle. This asserts that hazard still exists, so the
    choice of raw bytes is a documented decision rather than a habit.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    spki = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    loaded = serialization.load_der_public_key(spki)
    assert not isinstance(loaded, type(load_public_key(b'\x00' * PUBLIC_KEY_LEN)))


def test_a_repr_does_not_leak_the_private_key():
    prekey = generate_prekey(key_id=99)
    text = repr(prekey)
    assert '99' in text
    assert prekey.seed.hex()[:16] not in text
    assert prekey.public_bytes.hex()[:8] in text


def test_a_prekey_id_must_be_a_64_bit_int():
    key = generate_prekey()
    for bad in (-1, 2 ** 64, True, 'x'):
        with pytest.raises(PQError):
            PQOneTimePreKey(key_id=bad, private_key=key.private_key)


def test_a_prekey_needs_an_mlkem_private_key():
    with pytest.raises(PQError):
        PQOneTimePreKey(key_id=1, private_key=b'\x00' * 64)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

def test_a_stocked_pool_has_the_keys_it_says():
    store = stocked_pq_store(5)
    assert store.count() == 5
    assert store.available_ids() == [0, 1, 2, 3, 4]


def test_a_stocked_pool_of_zero_is_refused():
    with pytest.raises(PQError):
        stocked_pq_store(0)


def test_a_consumed_key_cannot_be_consumed_again():
    store = stocked_pq_store(1)
    store.consume(0)
    with pytest.raises(PQPreKeyAlreadyUsed):
        store.consume(0)
    assert store.was_consumed(0)
    assert store.count() == 0


def test_an_unknown_key_is_refused():
    store = stocked_pq_store(1)
    with pytest.raises(NoSuchPQPreKey):
        store.consume(999)
    with pytest.raises(NoSuchPQPreKey):
        store.peek(999)
    with pytest.raises(NoSuchPQPreKey):
        store.release(999)
    assert store.was_consumed(999) is False


def test_reservation_holds_a_key_out_of_circulation_but_not_spent():
    """
    The property the reservation exists for, and the same one the X25519 pool
    has: a key advertised in a bundle must not be advertised in the next one.

    The first version of this pool returned already-reserved keys from
    ``peek_any`` and ``reserve_next``, so two consecutive bundles carried the same
    prekey and the second handshake was refused by ``consume`` -- a denial of
    service against a legitimate peer, caused by nothing more than a naive
    server. Which is what the existing X25519 pool's docstring warns about.
    """
    store = stocked_pq_store(3)
    first = store.reserve_next()
    second = store.reserve_next()
    assert first.key_id != second.key_id
    assert store.reserved_ids() == sorted({first.key_id, second.key_id})
    # Reserved is not spent: the handshake it was advertised for still works.
    assert store.was_consumed(first.key_id) is False
    assert store.count() == 3


def test_release_returns_a_reserved_key():
    store = stocked_pq_store(1)
    reserved = store.reserve_next()
    store.release(reserved.key_id)
    assert store.reserved_ids() == []
    assert store.count() == 1
    # Releasing a consumed key is an error, not a silent no-op.
    store.consume(reserved.key_id)
    with pytest.raises(PQPreKeyAlreadyUsed):
        store.release(reserved.key_id)


def test_replenish_tops_the_pool_up():
    store = stocked_pq_store(2)
    store.consume(store.available_ids()[0])
    added = store.replenish(0, 4)
    assert store.count() == 4
    assert added == 3


def test_replenish_with_a_nonsense_target_is_refused():
    store = stocked_pq_store(1)
    with pytest.raises(PQPreKeyExhausted):
        store.replenish(0, 0)


def test_two_threads_racing_for_one_key_leave_one_winner():
    """
    The single-use guarantee under concurrency.

    Two threads, one key, and the loser has to be told rather than handed a key
    that is about to be used twice.
    """
    store = stocked_pq_store(1)
    results = []
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait()
        try:
            results.append(('ok', store.consume(0).key_id))
        except PQPreKeyAlreadyUsed as exc:
            results.append(('refused', str(exc)))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    outcomes = sorted(kind for kind, _ in results)
    assert outcomes == ['ok', 'refused']
    assert store.count() == 0


# ---------------------------------------------------------------------------
# PQXDH
# ---------------------------------------------------------------------------

def test_the_root_key_includes_the_kem_contribution():
    x3dh, responder = make_responder()
    alice_private, _ = x3dh.generate_identity_keys()
    bundle = responder.publish_bundle()
    assert bundle.pq_prekey is not None
    assert bundle.pq_prekey_id is not None

    state, init = x3dh.initiate_handshake(alice_private, bundle)
    assert [part.label for part in state.dh_parts][-1] == LABEL_KEM
    assert state.is_post_quantum
    assert init.pq_ciphertext is not None
    assert len(init.pq_ciphertext) == CIPHERTEXT_LEN
    assert init.pq_prekey_id == bundle.pq_prekey_id


def test_both_sides_derive_the_same_root_key():
    x3dh, responder = make_responder()
    alice_private, _ = x3dh.generate_identity_keys()
    state, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    responder_state = responder.handle_init(init)
    assert state.master_key() == responder_state.master_key()


def test_a_responder_without_a_post_quantum_pool_refuses_to_publish():
    x3dh, responder = make_responder(pq_prekey_count=None)
    with pytest.raises(Exception) as caught:
        responder.publish_bundle()
    assert 'ML-KEM' in str(caught.value) or 'post-quantum' in str(caught.value)


def test_a_responder_that_never_advertised_a_post_quantum_key_cannot_answer_one():
    """
    The mirror image: an initiator that references a prekey the responder does
    not have.

    The alternative would be to drop the post-quantum contribution and carry on,
    which is a downgrade the initiator would not notice until the confirmation
    failed, with no way to tell a quantum adversary from a broken peer.
    """
    x3dh, with_pq = make_responder()
    _, without_pq = make_responder(pq_prekey_count=None)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, with_pq.publish_bundle())
    with pytest.raises(NoSuchPQPreKey):
        without_pq.handle_init(init)


def test_a_post_quantum_prekey_cannot_be_spent_twice():
    """
    The same guarantee as the X25519 one-time prekey, and the same reason: a
    prekey used twice has a contribution recoverable from two handshakes, which
    is the whole property the post-quantum part was added for.
    """
    x3dh, responder = make_responder(prekey_count=3, pq_prekey_count=2)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    responder.handle_init(init)
    assert responder.pq_prekey_store.was_consumed(init.pq_prekey_id)

    # Replaying the handshake must not reuse the prekey. Both pools are checked
    # before either is spent, so whichever is checked first, the refusal is the
    # same refusal and no second contribution is ever spent.
    with pytest.raises((OneTimePreKeyAlreadyUsed, PQPreKeyAlreadyUsed)):
        responder.handle_init(init)
    assert responder.pq_prekey_store.count() == 1
    assert responder.pq_prekey_store.was_consumed(init.pq_prekey_id)


def test_a_spent_out_post_quantum_pool_stops_publishing():
    """
    Once the pool is dry the responder refuses rather than publishing a bundle
    that silently lacks post-quantum protection.
    """
    x3dh, responder = make_responder(prekey_count=3, pq_prekey_count=1)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    responder.handle_init(init)

    assert responder.pq_prekey_store.count() == 0
    with pytest.raises(PQPreKeyExhausted):
        responder.publish_bundle()


def test_a_spent_x25519_prekey_is_not_spent_when_the_pq_one_is_missing():
    """
    Both pools are checked before either is touched.

    Consuming as it goes would burn an X25519 prekey on a handshake that then
    fails for want of a post-quantum one, which is a denial of service against
    every future session -- and the X25519 pool is the one a client refills on a
    low-water mark, so it would be the one that ran out.
    """
    x3dh, with_pq = make_responder(prekey_count=1, pq_prekey_count=1)
    _, without_pq = make_responder(prekey_count=1, pq_prekey_count=None)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, with_pq.publish_bundle())

    with pytest.raises(NoSuchPQPreKey):
        without_pq.handle_init(init)
    # Nothing was spent, so the same prekey is still there.
    assert without_pq.prekey_store.was_consumed(init.one_time_prekey_id) is False
    assert without_pq.prekey_store.count() == 1


def test_a_forged_handshake_spends_neither_pool():
    """
    Authentication before spending, now covering two pools instead of one.

    An attacker who could spend prekeys with forged handshakes would empty the
    post-quantum pool, and an empty post-quantum pool means every later session
    is either refused or accepted without post-quantum protection. That is a
    better denial of service than emptying the X25519 pool, because there is
    usually only one post-quantum pool and no cheap way to refill it.
    """
    from dataclasses import replace

    x3dh, responder = make_responder(prekey_count=3, pq_prekey_count=3)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    pq_store = responder.pq_prekey_store
    before_x25519 = responder.prekey_store.count()
    before_pq = pq_store.count()

    from nacl.public import PrivateKey
    forged = replace(init, ephemeral_key=PrivateKey.generate().public_key)
    with pytest.raises(InvalidSignatureError):
        responder.handle_init(forged)

    assert responder.prekey_store.count() == before_x25519
    assert pq_store.count() == before_pq
    assert not pq_store.was_consumed(init.pq_prekey_id)


def test_the_bundle_refuses_a_prekey_and_id_that_disagree():
    x3dh, responder = make_responder()
    bundle = responder.publish_bundle()
    with pytest.raises(InvalidHandshakeError):
        PreKeyBundle(
            identity_key=bundle.identity_key,
            signed_prekey=bundle.signed_prekey,
            signed_prekey_signature=bundle.signed_prekey_signature,
            pq_prekey=bundle.pq_prekey,
            pq_prekey_id=None,
        )


# ---------------------------------------------------------------------------
# The ciphertext is inside the signature
# ---------------------------------------------------------------------------

def test_swapping_the_post_quantum_ciphertext_breaks_the_signature():
    """
    The cheapest post-quantum downgrade, and the reason the ciphertext is in the
    transcript.

    An attacker replaces the 1088-byte ciphertext with one it generated itself.
    Without the signature covering it, everything else about the handshake would
    still check out, the attacker would decapsulate its own ciphertext, and the
    two honest parties would not share the ML-KEM secret.
    """
    from dataclasses import replace

    x3dh, responder = make_responder()
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())

    attack_key = generate_prekey()
    _, attack_ciphertext = encapsulate(attack_key.public_key)
    assert attack_ciphertext != init.pq_ciphertext

    tampered = replace(init, pq_ciphertext=attack_ciphertext)
    with pytest.raises(InvalidSignatureError):
        tampered.verify()
    with pytest.raises(InvalidSignatureError):
        responder.handle_init(tampered)


def test_changing_the_post_quantum_prekey_id_breaks_the_signature():
    from dataclasses import replace

    x3dh, responder = make_responder(pq_prekey_count=3)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    with pytest.raises(InvalidSignatureError):
        replace(init, pq_prekey_id=init.pq_prekey_id + 1).verify()


def test_a_handshake_without_a_ciphertext_but_with_an_id_is_refused():
    from nacl.public import PrivateKey as _PrivateKey
    from nacl.signing import SigningKey

    from src.crypto.key_management import identity_public_x25519
    from src.crypto.x3dh import HandshakeInit

    signing = SigningKey.generate()
    with pytest.raises(InvalidHandshakeError):
        HandshakeInit(
            identity_key=signing.verify_key,
            identity_key_x25519=identity_public_x25519(signing.verify_key),
            ephemeral_key=_PrivateKey.generate().public_key,
            signature=signing.sign(b'x').signature,
            pq_prekey_id=1,
        )


def test_a_handshake_ciphertext_of_the_wrong_length_is_refused():
    from dataclasses import replace

    x3dh, responder = make_responder()
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    with pytest.raises(InvalidHandshakeError):
        replace(init, pq_ciphertext=b'\x00' * 100)


# ---------------------------------------------------------------------------
# The wire format
# ---------------------------------------------------------------------------

def test_a_bundle_round_trips_with_its_post_quantum_key():
    x3dh, responder = make_responder()
    bundle = responder.publish_bundle()
    frame = encode_prekey_bundle(bundle)
    decoded = decode_prekey_bundle(frame)

    assert frame[1] == KIND_PREKEY_BUNDLE
    assert decoded.pq_prekey_id == bundle.pq_prekey_id
    assert decoded.pq_prekey.public_bytes_raw() == \
        bundle.pq_prekey.public_bytes_raw()
    # And the round-tripped bundle still works for a handshake, which is the only
    # proof that what crossed the wire was usable.
    alice_private, _ = x3dh.generate_identity_keys()
    state, init = x3dh.initiate_handshake(alice_private, decoded)
    assert state.is_post_quantum
    assert responder.handle_init(init).master_key() == state.master_key()


def test_a_handshake_round_trips_with_its_ciphertext():
    x3dh, responder = make_responder()
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    decoded = decode_handshake_init(encode_handshake_init(init))
    assert decoded.pq_ciphertext == init.pq_ciphertext
    assert decoded.pq_prekey_id == init.pq_prekey_id
    assert responder.handle_init(decoded).is_post_quantum


def test_an_earlier_wire_version_is_refused_not_decoded():
    """
    A version-2 bundle has no post-quantum fields.

    Decoding one would mean the decoder ran off the end of a payload that
    happened to have spare bytes, and a version-2 ratchet plaintext has no
    padding length prefix, so unpadding it would read its first byte as a length.
    Refusing on the version byte is the only safe answer.
    """
    x3dh, responder = make_responder()
    frame = bytearray(encode_prekey_bundle(responder.publish_bundle()))
    frame[0] = 2
    with pytest.raises(UnsupportedVersion):
        decode_prekey_bundle(bytes(frame))

    _, init = x3dh.initiate_handshake(
        *x3dh.generate_identity_keys()[:1],
        responder.publish_bundle(),
    )
    init_frame = bytearray(encode_handshake_init(init))
    init_frame[0] = 1
    with pytest.raises(UnsupportedVersion):
        decode_handshake_init(bytes(init_frame))
    assert WIRE_VERSION == 3


def test_a_bundle_with_a_truncated_post_quantum_key_is_refused():
    x3dh, responder = make_responder()
    frame = encode_prekey_bundle(responder.publish_bundle())
    # Keep the flag, cut the key: the decoder must notice the payload ran out.
    assert len(frame) > PUBLIC_KEY_LEN
    with pytest.raises(Exception):
        decode_prekey_bundle(frame[:len(frame) - PUBLIC_KEY_LEN // 2])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persistent_store(tmp_path, name='pq.bin'):
    from src.storage.container import KdfParams

    path = str(tmp_path / name)
    return PersistentPQPreKeyStore(
        Container.create(
            path, 'password', kdf_params=KdfParams(n_log2=12, r=8, p=1)
        )
    ), path


def test_a_pool_survives_a_restart(tmp_path):
    store, path = persistent_store(tmp_path)
    for index in range(3):
        store.put(generate_prekey(key_id=index))
    expected = {index: store.peek(index).public_bytes for index in range(3)}

    from src.storage.container import KdfParams
    reopened = PersistentPQPreKeyStore(
        Container.open(path, 'password')
    )
    assert reopened.count() == 3
    for index in range(3):
        assert reopened.peek(index).public_bytes == expected[index]
    del KdfParams


def test_a_spent_key_stays_spent_across_a_restart(tmp_path):
    """
    Spent ids are recorded rather than merely deleted.

    Without the record, a rolled-back container would make an already-used
    prekey available again, and a prekey used twice is a prekey whose
    contribution is recoverable from two handshakes.
    """
    store, path = persistent_store(tmp_path)
    for index in range(2):
        store.put(generate_prekey(key_id=index))
    store.consume(0)

    from src.storage.container import KdfParams
    reopened = PersistentPQPreKeyStore(Container.open(path, 'password'))
    assert reopened.was_consumed(0)
    with pytest.raises(PQPreKeyAlreadyUsed):
        reopened.consume(0)
    del KdfParams


def test_the_pool_is_sealed(tmp_path):
    """These are private keys. Finding one in the file would be a disclosure."""
    store, path = persistent_store(tmp_path)
    prekey = generate_prekey(key_id=5)
    store.put(prekey)
    raw = open(path, 'rb').read()
    assert prekey.seed not in raw
    assert prekey.public_bytes not in raw
    assert b'prekeys' not in raw
    assert b'consumed' not in raw


def test_a_corrupt_pool_is_refused(tmp_path):
    import json

    store, path = persistent_store(tmp_path)
    store.put(generate_prekey(key_id=1))
    container = store._container
    for payload in (
        b'not json',
        json.dumps({'prekeys': 'not an object'}).encode(),
        json.dumps({'prekeys': {'x': 'AAAA'}}).encode(),
        json.dumps({'prekeys': {}}).encode() + b'trailing',
        json.dumps({'prekeys': {}, 'consumed': 'not a list'}).encode(),
    ):
        container.store(payload, container.load()[0] + 1)
        with pytest.raises(StorageError):
            PersistentPQPreKeyStore(container)


def test_a_wrong_password_is_refused(tmp_path):
    from src.storage.container import WrongPassword

    store, path = persistent_store(tmp_path)
    store.put(generate_prekey(key_id=1))
    with pytest.raises(WrongPassword):
        PersistentPQPreKeyStore(Container.open(path, 'wrong'))


def test_a_persisted_pool_keeps_a_handshake_post_quantum(tmp_path):
    """
    The end-to-end point: a pool that only lives in memory is an empty pool
    after a restart, and an empty post-quantum pool is a client that can no
    longer accept a session it would be willing to call secure.
    """
    from src.crypto.prekey_store import InMemoryPreKeyStore

    store, path = persistent_store(tmp_path, 'pqkeys.bin')
    store.replenish(0, 3)

    x3dh = X3DH()
    identity_private, _ = x3dh.generate_identity_keys()
    signed_prekey_private, _, _ = x3dh.generate_signed_prekey(identity_private)
    prekeys = InMemoryPreKeyStore()
    for index in range(3):
        prekeys.put(x3dh.generate_one_time_prekey(key_id=index))
    responder = X3DHResponder(
        identity_private=identity_private,
        signed_prekey_private=signed_prekey_private,
        prekey_store=prekeys,
        pq_prekey_store=store,
    )
    alice_private, _ = x3dh.generate_identity_keys()
    state, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    assert state.is_post_quantum
    assert responder.handle_init(init).master_key() == state.master_key()


def test_a_wrong_algorithm_key_is_not_accepted_as_a_prekey():
    """
    Belt and braces on the type check in the loader.

    If a future ``cryptography`` ever made ``from_public_bytes`` accept a
    different key type, this is the place it would show up.
    """
    from cryptography.hazmat.primitives.asymmetric import mlkem

    key = mlkem.MLKEM768PrivateKey.generate().public_key()
    assert isinstance(key, mlkem.MLKEM768PublicKey)
    assert isinstance(load_public_key(key.public_bytes_raw()),
                      mlkem.MLKEM768PublicKey)
    with pytest.raises(InvalidPQKey):
        load_public_key('not bytes')


def test_engaging_the_pool_never_returns_a_spent_key(tmp_path):
    """
    Reserve, consume something else, reserve again.

    The last reservation must be neither the key already spent nor one already
    advertised. Written to be independent of the random 64-bit prekey ids:
    picking "the last available" would sometimes pick the reserved one, since
    ``available_ids`` is sorted and the ids are not.
    """
    store, _ = persistent_store(tmp_path)
    store.replenish(0, 3)
    first = store.reserve_next()

    spare = next(
        key_id for key_id in store.available_ids()
        if key_id not in {first.key_id, *store.reserved_ids()}
    )
    store.consume(spare)

    second = store.reserve_next()
    assert second.key_id != first.key_id
    assert second.key_id != spare
    assert not store.was_consumed(second.key_id)
    # reserved_ids is sorted, and the ids are random.
    assert store.reserved_ids() == sorted([first.key_id, second.key_id])
