"""
Tests for the one-to-one session: the bridge from X3DH to the Double Ratchet.

Before this module existed, no code in the repository constructed a
``DoubleRatchet`` at all. Every test assembled one by hand around a constant
root key, which both sides derived identically -- so the suite was green while
nothing connected the handshake to the ratchet. These tests are the ones that
would have caught that: every message here travels the real path
``Session.send`` -> ``wire.encode_*`` -> bytes -> ``wire.decode_*`` ->
``Session.receive``, and every key comes from a real handshake.

Two properties get the most attention:

1. The initiator cannot use the session before the responder's key
   confirmation verifies. A root key nobody has proved they share is exactly
   the value a MITM wants, and a ratchet built over it looks locally correct
   right up to the first message the peer cannot read.
2. The associated data binds a message to *this* session. Without it, a valid
   ciphertext is only bound to its own header, and a frame that is valid
   somewhere else is valid nowhere in particular.
"""

import pytest
from nacl.public import PublicKey

from src.crypto.double_ratchet import (
    PROTOCOL_VERSION,
    DecryptionError,
    DoubleRatchet,
    IncompatibleSessionError,
    InvalidSessionStateError,
)
from src.crypto.prekey_store import (
    InMemoryPreKeyStore,
    OneTimePreKeyAlreadyUsed,
    make_one_time_prekey,
)
from src.crypto.x3dh import (
    X3DH,
    InvalidKeyError,
    InvalidSignatureError,
    KeyConfirmationFailed,
    KeyConfirmationRequired,
    PreKeyBundle,
    RetiredSignedPrekey,
    X3DHResponder,
)
from src.protocol.session import (
    Session,
    SessionError,
    accept,
    initiate,
    session_associated_data,
)
from src.protocol.wire import (
    KIND_HANDSHAKE_INIT,
    KIND_KEY_CONFIRMATION,
    KIND_PREKEY_BUNDLE,
    KIND_RATCHET_MESSAGE,
    UnknownMessageKind,
    encode_prekey_bundle,
)

PREKEYS = 5


def make_responder(prekeys: int = PREKEYS, signed_prekey_id: int = 1):
    """A responder with a full prekey pool, plus its identity key pair."""
    x3dh = X3DH()
    identity_private, identity_public = x3dh.generate_identity_keys()
    signed_prekey_private, signed_prekey_public, signature = (
        x3dh.generate_signed_prekey(identity_private)
    )
    store = InMemoryPreKeyStore()
    for key_id in range(prekeys):
        store.put(make_one_time_prekey(key_id=key_id))
    responder = X3DHResponder(
        identity_private=identity_private,
        signed_prekey_private=signed_prekey_private,
        prekey_store=store,
        signed_prekey_id=signed_prekey_id,
    )
    return responder, identity_public, store


def handshake(prekeys: int = PREKEYS, **kwargs):
    """
    Run a complete handshake and return both usable sessions.

    Every test that wants a working conversation goes through here, so a change
    in the handshake ordering breaks them all at once rather than one at a time.
    """
    responder, responder_public, store = make_responder(prekeys)
    bundle = responder.publish_bundle()
    initiator_private, initiator_public = X3DH().generate_identity_keys()
    initiator, init_frame = initiate(initiator_private, bundle, **kwargs)
    responder_session, confirmation = accept(responder, init_frame)
    initiator.confirm(confirmation)
    return {
        'initiator': initiator,
        'responder': responder_session,
        'initiator_private': initiator_private,
        'initiator_public': initiator_public,
        'responder_public': responder_public,
        'store': store,
    }


# ---------------------------------------------------------------------------
# The full cycle
# ---------------------------------------------------------------------------

def test_handshake_produces_the_frames_the_specification_calls_for():
    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()

    _, init_frame = initiate(initiator_private, bundle)
    assert init_frame[1] == KIND_HANDSHAKE_INIT

    _, confirmation = accept(responder, init_frame)
    assert confirmation[1] == KIND_KEY_CONFIRMATION


def test_a_bundle_is_a_bundle_frame():
    responder, _, _ = make_responder()
    assert encode_prekey_bundle(responder.publish_bundle())[1] == \
        KIND_PREKEY_BUNDLE


def test_messages_flow_both_ways_through_the_wire():
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']

    frame = initiator.send(b'privet')
    assert frame[1] == KIND_RATCHET_MESSAGE
    assert responder_session.receive(frame) == b'privet'

    assert initiator.receive(responder_session.send(b'privet, kak dela')) == \
        b'privet, kak dela'


def test_a_long_conversation_in_both_directions():
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']
    for index in range(25):
        sent = b'message %d' % index
        assert responder_session.receive(initiator.send(sent)) == sent
        reply = b'reply %d' % index
        assert initiator.receive(responder_session.send(reply)) == reply


def test_empty_and_large_messages():
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']
    assert responder_session.receive(initiator.send(b'')) == b''
    big = bytes(range(256)) * 400  # 102400 bytes
    assert responder_session.receive(initiator.send(big)) == big


def test_out_of_order_delivery():
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']
    first, second, third = (initiator.send(b'1'), initiator.send(b'2'),
                            initiator.send(b'3'))
    assert responder_session.receive(third) == b'3'
    assert responder_session.receive(first) == b'1'
    assert responder_session.receive(second) == b'2'


def test_the_sender_turns_over_when_its_messages_run_out_first():
    """
    The DH ratchet, not the session, decides who ratchets.

    If the responder answers more messages than it sent, it has to perform a
    DH ratchet of its own and re-key. A session that assumed a fixed turn
    order would desynchronise here instead of recovering.
    """
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']
    # The first message has to arrive before the responder can answer at all:
    # its remote ratchet key comes out of that header.
    assert responder_session.receive(initiator.send(b'one')) == b'one'
    for index in range(4):
        assert initiator.receive(
            responder_session.send(b'reply %d' % index)
        ) == b'reply %d' % index
    assert responder_session.receive(initiator.send(b'two')) == b'two'


def test_the_responder_cannot_send_before_it_has_heard_anything():
    """
    The specification gives the responder no remote ratchet key to start with.

    It has to learn the initiator's key from the first header received, so
    sending first is not a turn-order mistake to be tolerated -- there is no
    correct key to encrypt under. Failing here is the specification working,
    not a gap in it.
    """
    pair = handshake()
    with pytest.raises(InvalidSessionStateError):
        pair['responder'].send(b'out of turn')


def test_the_handshake_spends_a_one_time_prekey():
    pair = handshake()
    spent = pair['initiator'].one_time_prekey_id
    assert spent is not None
    assert pair['store'].was_consumed(spent)


# ---------------------------------------------------------------------------
# The confirmation gate
# ---------------------------------------------------------------------------

def test_the_initiator_cannot_send_before_confirming():
    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    initiator, _ = initiate(initiator_private, bundle)

    assert not initiator.is_confirmed
    with pytest.raises(KeyConfirmationRequired):
        initiator.send(b'too early')
    with pytest.raises(KeyConfirmationRequired):
        initiator.state


def test_the_responder_needs_no_confirmation():
    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    _, init_frame = initiate(initiator_private, bundle)
    responder_session, _ = accept(responder, init_frame)
    # It authenticated the initiator and computed the secret itself, so there
    # is nothing left for it to verify.
    assert responder_session.is_confirmed


def test_a_forged_confirmation_is_refused_and_stays_refused():
    """
    A MAC that does not verify must leave the session locked.

    The temptation is to clear the flag and let the caller retry, which would
    turn a failed confirmation into a second attempt at using a root key the
    peer never proved it shares.
    """
    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    initiator, init_frame = initiate(initiator_private, bundle)

    # A confirmation from an unrelated handshake: a valid frame, wrong MAC.
    other_responder, _, _ = make_responder()
    other_initiator_private, _ = X3DH().generate_identity_keys()
    _, other_init_frame = initiate(
        other_initiator_private, other_responder.publish_bundle()
    )
    _, foreign_confirmation = accept(other_responder, other_init_frame)

    with pytest.raises(KeyConfirmationFailed):
        initiator.confirm(foreign_confirmation)
    assert not initiator.is_confirmed
    with pytest.raises(KeyConfirmationRequired):
        initiator.send(b'still locked')
    # Retrying the same bad confirmation fails the same way; it is not quietly
    # accepted on the second pass.
    with pytest.raises(KeyConfirmationFailed):
        initiator.confirm(foreign_confirmation)


def test_confirming_on_the_responder_side_is_refused():
    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    _, init_frame = initiate(initiator_private, bundle)
    responder_session, confirmation = accept(responder, init_frame)
    with pytest.raises(SessionError):
        responder_session.confirm(confirmation)


def test_confirming_twice_is_refused():
    pair_responder, _, _ = make_responder()
    initiator_private, _ = X3DH().generate_identity_keys()
    initiator, init_frame = initiate(
        initiator_private, pair_responder.publish_bundle()
    )
    _, confirmation = accept(pair_responder, init_frame)
    initiator.confirm(confirmation)
    initiator.send(b'ok')
    with pytest.raises(SessionError):
        initiator.confirm(confirmation)


# ---------------------------------------------------------------------------
# Associated data
# ---------------------------------------------------------------------------

def test_both_sides_derive_the_same_associated_data():
    pair = handshake()
    assert pair['initiator'].associated_data == pair['responder'].associated_data


def test_associated_data_binds_both_identities_and_the_version():
    _, alice = X3DH().generate_identity_keys()
    _, bob = X3DH().generate_identity_keys()
    _, carol = X3DH().generate_identity_keys()

    base = session_associated_data(alice, bob)
    # Order is part of the value: the two sides agree on it, nobody chooses it
    # per call.
    assert session_associated_data(bob, alice) != base
    assert session_associated_data(alice, carol) != base
    assert session_associated_data(carol, bob) != base
    assert session_associated_data(alice, bob, PROTOCOL_VERSION) == base
    assert session_associated_data(alice, bob, PROTOCOL_VERSION + 1) != base


def test_the_associated_data_encodes_the_role_through_the_key_order():
    """
    Which side comes first *is* the role, and both sides agree on it.

    So the associated data does depend on the role, and stays consistent
    because the initiator and the responder each know who started. This is why
    swapping the two identities when restoring is caught, and why the role flag
    on its own is not (see the next test).
    """
    pair = handshake()
    initiator = pair['initiator']

    assert session_associated_data(
        pair['initiator_public'], pair['responder_public']
    ) == initiator.associated_data
    # The same two keys in the other order is a different value.
    assert session_associated_data(
        pair['responder_public'], pair['initiator_public']
    ) != initiator.associated_data


def test_associated_data_rejects_non_keys():
    _, alice = X3DH().generate_identity_keys()
    with pytest.raises(SessionError):
        session_associated_data(bytes(alice), alice)
    with pytest.raises(SessionError):
        session_associated_data(alice, None)


def test_a_frame_from_a_different_session_does_not_decrypt():
    """The reason the associated data is derived rather than left empty."""
    pair = handshake()
    initiator = pair['initiator']
    frame = initiator.send(b'for this conversation only')

    # Same ratchet, same key schedule, different associated data: exactly the
    # case a replay across sessions would produce.
    impostor = Session.from_state(
        initiator.state,
        is_initiator=True,
        local_identity=pair['initiator_public'],
        peer_identity=pair['responder_public'],
    )
    with pytest.raises(DecryptionError):
        impostor.receive(frame)


def test_identities_are_reported_from_the_right_side():
    pair = handshake()
    assert pair['initiator'].local_identity == pair['initiator_public']
    assert pair['initiator'].peer_identity == pair['responder_public']
    assert pair['responder'].local_identity == pair['responder_public']
    assert pair['responder'].peer_identity == pair['initiator_public']


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_a_restored_session_continues_the_conversation():
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']
    # A message in flight across the restart, which the restored pair has to
    # pick up from the middle.
    in_flight = initiator.send(b'before the restart')
    assert responder_session.receive(in_flight) == b'before the restart'

    revived_initiator = Session.from_state(
        initiator.state,
        is_initiator=True,
        local_identity=pair['initiator_public'],
        peer_identity=pair['responder_public'],
    )
    revived_responder = Session.from_state(
        responder_session.state,
        is_initiator=False,
        local_identity=pair['responder_public'],
        peer_identity=pair['initiator_public'],
    )
    assert revived_responder.receive(revived_initiator.send(b'after')) == b'after'
    assert revived_initiator.receive(revived_responder.send(b'reply')) == b'reply'


def test_a_restored_session_keeps_its_associated_data():
    pair = handshake()
    revived = Session.from_state(
        pair['initiator'].state,
        is_initiator=True,
        local_identity=pair['initiator_public'],
        peer_identity=pair['responder_public'],
    )
    assert revived.associated_data == pair['initiator'].associated_data
    # Restoring is not a handshake, so there is nothing left to confirm.
    assert revived.is_confirmed
    assert revived.one_time_prekey_id is None


def test_a_state_that_cannot_belong_to_the_claimed_role_is_refused():
    """
    An initiator ratchets the moment it is built, so its state always carries a
    remote ratchet key. A state without one was written by a responder that had
    not received anything, and claiming otherwise is provably wrong.
    """
    responder, responder_public, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, initiator_public = X3DH().generate_identity_keys()
    _, init_frame = initiate(initiator_private, bundle)
    responder_session, _ = accept(responder, init_frame)
    virgin = responder_session.state.serialize()
    assert virgin['dh_remote_pub'] is None

    with pytest.raises(SessionError):
        Session.from_state(
            virgin,
            is_initiator=True,
            local_identity=initiator_public,
            peer_identity=responder_public,
        )


def test_swapping_the_two_identities_when_restoring_is_caught_loudly():
    """
    Handing the identities over the wrong way round changes the associated
    data, so every message fails at the AEAD instead of going to the wrong
    file.
    """
    pair = handshake()
    initiator = pair['initiator']
    document = initiator.state.serialize()
    frame = initiator.send(b'a message')

    swapped = Session.from_state(
        document,
        is_initiator=True,
        local_identity=pair['responder_public'],
        peer_identity=pair['initiator_public'],
    )
    assert swapped.associated_data != initiator.associated_data
    with pytest.raises(DecryptionError):
        swapped.receive(frame)


def test_the_role_alone_is_enough_to_break_the_associated_data():
    """
    One wrong argument is caught, because the role fixes which identity comes
    first in the associated data.

    ``from_state`` reorders the two keys according to the claimed role, so
    claiming the wrong role for a correctly passed pair of identities produces
    a different associated data and every message fails at the AEAD.
    """
    pair = handshake()
    responder_session = pair['responder']
    # One message has to arrive first, otherwise the responder's state still has
    # no remote ratchet key and the separate provable check in from_state fires
    # first.
    assert responder_session.receive(pair['initiator'].send(b'first')) == b'first'
    document = responder_session.state.serialize()
    frame = pair['initiator'].send(b'a message')

    mislabelled = Session.from_state(
        document,
        is_initiator=True,
        local_identity=pair['responder_public'],
        peer_identity=pair['initiator_public'],
    )
    assert mislabelled.associated_data != responder_session.associated_data
    with pytest.raises(DecryptionError):
        mislabelled.receive(frame)


def test_a_wrong_role_compensated_by_swapped_identities_is_not_caught():
    """
    The one failure this design does not catch, stated as a test.

    Two wrong arguments can cancel: claim the wrong role *and* hand the
    identities over the wrong way round, and the associated data comes out
    right again. The ratchet has no notion of a role at all -- it has a sending
    chain and a receiving chain -- so it decrypts, and the only thing wrong is
    which identity is reported as the peer's. The cure is the caller's care.
    Pretending otherwise would hide the gap rather than close it.
    """
    pair = handshake()
    responder_session = pair['responder']
    assert responder_session.receive(pair['initiator'].send(b'first')) == b'first'
    document = responder_session.state.serialize()
    frame = pair['initiator'].send(b'a message')

    mislabelled = Session.from_state(
        document,
        is_initiator=True,
        local_identity=pair['initiator_public'],
        peer_identity=pair['responder_public'],
    )
    assert mislabelled.associated_data == responder_session.associated_data
    assert mislabelled.receive(frame) == b'a message'
    # This side is the responder; the restored session calls the initiator's key
    # its own.
    assert mislabelled.local_identity == pair['initiator_public']
    assert mislabelled.local_identity == pair['responder'].peer_identity


def test_from_state_refuses_a_state_from_another_protocol_version():
    pair = handshake()
    document = pair['initiator'].state.serialize()
    document['version'] = PROTOCOL_VERSION - 1
    with pytest.raises(IncompatibleSessionError):
        Session.from_state(
            document,
            is_initiator=True,
            local_identity=pair['initiator_public'],
            peer_identity=pair['responder_public'],
        )


def test_a_session_needs_exactly_one_way_to_build_its_ratchet():
    pair = handshake()
    with pytest.raises(SessionError):
        Session(
            is_initiator=True,
            initiator_identity=pair['initiator_public'],
            responder_identity=pair['responder_public'],
        )
    with pytest.raises(SessionError):
        Session(
            is_initiator=True,
            ratchet=DoubleRatchet.from_state(pair['initiator'].state),
            ratchet_factory=lambda: None,
            initiator_identity=pair['initiator_public'],
            responder_identity=pair['responder_public'],
        )


def test_a_role_that_contradicts_its_handshake_is_refused():
    """
    ``is_initiator`` and the handshake it came with have to agree.

    They are passed separately so a restored session can state its role, which
    means a caller can hand over a role that contradicts the handshake it also
    handed over. The ratchet keys would still be right, so nothing else would
    complain: the failure would appear later as messages that never decrypt.
    """
    responder, responder_public, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, initiator_public = X3DH().generate_identity_keys()
    x3dh_session, _ = X3DH().begin(initiator_private, bundle)

    with pytest.raises(SessionError):
        Session(
            is_initiator=False,
            ratchet_factory=lambda: None,
            x3dh_session=x3dh_session,
            initiator_identity=initiator_public,
            responder_identity=responder_public,
        )


# ---------------------------------------------------------------------------
# Failures at the handshake boundary
# ---------------------------------------------------------------------------

def test_a_tampered_bundle_never_reaches_the_ratchet():
    """The signature check happens before any bundle key is used."""
    responder, responder_public, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()

    forged = PublicKey(bytes(bundle.signed_prekey))
    forged = PublicKey(
        (bytes(forged)[:-1] + bytes([bytes(forged)[-1] ^ 1]))
    )
    tampered = PreKeyBundle(
        identity_key=bundle.identity_key,
        signed_prekey=forged,
        signed_prekey_signature=bundle.signed_prekey_signature,
        one_time_prekey=bundle.one_time_prekey,
        one_time_prekey_id=bundle.one_time_prekey_id,
        signed_prekey_id=bundle.signed_prekey_id,
    )
    with pytest.raises(InvalidSignatureError):
        initiate(initiator_private, tampered)
    assert responder.identity_public == responder_public


def test_a_whole_bundle_substitution_is_possible_and_silent():
    """
    The one attack X3DH does not stop, pinned down as a test.

    An active attacker replaces the identity key *and* the signed prekey and
    signs the replacement with its own key. Every check the initiator performs
    passes, and it ends up in a real encrypted session with the attacker instead
    of the intended peer. Only out-of-band verification catches this, which is
    why docs/SECURITY_BOUNDARIES.md section 2 leads with it. A test that says so
    is worth more than a comment that says so.
    """
    responder, _, _ = make_responder()
    honest = responder.publish_bundle()

    evil_identity, evil_public = X3DH().generate_identity_keys()
    _, evil_spk_public, evil_signature = X3DH().generate_signed_prekey(
        evil_identity
    )
    substituted = PreKeyBundle(
        identity_key=evil_public,
        signed_prekey=evil_spk_public,
        signed_prekey_signature=evil_signature,
        one_time_prekey=honest.one_time_prekey,
        one_time_prekey_id=honest.one_time_prekey_id,
        signed_prekey_id=honest.signed_prekey_id,
    )

    initiator_private, _ = X3DH().generate_identity_keys()
    initiator, _ = initiate(initiator_private, substituted)

    # The signature verifies against the substituted identity key, which is the
    # whole point: it is a self-consistent bundle that is simply not the bundle
    # of the person being written to.
    X3DH.verify_bundle(substituted)
    assert initiator.peer_identity == evil_public
    assert initiator.peer_identity != responder.identity_public
    # And the session is usable, so nothing about the failure announces itself.
    assert initiator.is_confirmed is False
    with pytest.raises(KeyConfirmationRequired):
        initiator.send(b'to the wrong person')


def test_a_spent_prekey_ends_the_second_handshake():
    """
    Replaying a bundle cannot re-spend its one-time prekey.

    The initiator's signature covers the prekey id, so the responder cannot
    quietly fall back to the 3-DH variant: the two sides would derive different
    root keys and the initiator's confirmation check would fail with no useful
    diagnosis. Failing loudly at the prekey is the better trade.
    """
    responder, _, _ = make_responder(prekeys=1)
    bundle = responder.publish_bundle()
    first_private, _ = X3DH().generate_identity_keys()
    _, init_frame = initiate(first_private, bundle)
    accept(responder, init_frame)

    second_private, _ = X3DH().generate_identity_keys()
    _, second_init = initiate(second_private, bundle)
    with pytest.raises(OneTimePreKeyAlreadyUsed):
        accept(responder, second_init)


def test_a_retired_bundle_is_refused_when_the_caller_says_so():
    responder, _, _ = make_responder(signed_prekey_id=4)
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    with pytest.raises(RetiredSignedPrekey):
        initiate(initiator_private, bundle, minimum_signed_prekey_id=5)


def test_the_responder_will_not_accept_a_message_frame_as_a_handshake():
    pair = handshake()
    frame = pair['initiator'].send(b'not a handshake')
    responder, _, _ = make_responder()
    with pytest.raises(UnknownMessageKind):
        accept(responder, frame)


def test_a_session_refuses_a_non_bytes_message():
    pair = handshake()
    with pytest.raises(SessionError):
        pair['initiator'].send('a string is not bytes')


def test_the_specified_associated_data_still_reaches_the_handshake():
    """
    The handshake associated data is the caller's, and both sides must agree.

    It is a separate mechanism from the message-level associated data this
    module derives: this one is external context mixed into the transcript and
    the confirmation MAC, so a mismatch here shows up as a confirmation
    failure rather than as a message that will not decrypt.
    """
    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    initiator, init_frame = initiate(
        initiator_private, bundle, associated_data=b'conversation-42'
    )
    responder_session, confirmation = accept(
        responder, init_frame, associated_data=b'conversation-42'
    )
    initiator.confirm(confirmation)
    assert responder_session.receive(initiator.send(b'hello')) == b'hello'

    # Same keys, mismatched context: the confirmation must not verify.
    other, other_init = initiate(
        initiator_private, bundle, associated_data=b'conversation-43'
    )
    with pytest.raises(KeyConfirmationFailed):
        other.confirm(confirmation)


def test_a_bundle_carrying_a_low_order_signed_prekey_is_refused():
    """
    A degenerate X25519 key is caught by the DH step, not by the parser.

    The signature here is genuine, made with the identity key the bundle
    advertises, so the authentication check passes and the rejection has to come
    from the DH itself. That is the case worth covering: an all-zero public key
    produces an all-zero shared secret, which HKDF would turn into a root key
    both sides derive identically and an attacker holding no key at all also
    knows.
    """
    x3dh = X3DH()
    identity_private, identity_public = x3dh.generate_identity_keys()
    low_order = PublicKey(b'\x00' * 32)
    bundle = PreKeyBundle(
        identity_key=identity_public,
        signed_prekey=low_order,
        signed_prekey_signature=identity_private.sign(bytes(low_order)).signature,
        signed_prekey_id=1,
    )
    x3dh.verify_bundle(bundle)  # the signature is valid; the key is not usable

    initiator_private, _ = x3dh.generate_identity_keys()
    with pytest.raises(InvalidKeyError):
        initiate(initiator_private, bundle)


def test_private_key_types_do_not_leak_into_the_wire():
    """
    Nothing from the private side of either handshake reaches a frame.

    Cheap to state, expensive to lose: a private key that ends up in a frame is
    a frame anyone can decrypt.
    """
    pair = handshake()
    initiator, responder_session = pair['initiator'], pair['responder']
    secrets = [
        bytes(pair['initiator_private']),
        bytes(initiator.state.dh_local_priv),
        bytes(responder_session.state.dh_local_priv),
    ]
    # The responder cannot encrypt before it has heard the initiator's ratchet
    # key, so the first message has to be delivered.
    responder_session.receive(initiator.send(b'first'))
    for frame in (initiator.send(b'x'), responder_session.send(b'y')):
        for secret in secrets:
            assert secret not in frame
