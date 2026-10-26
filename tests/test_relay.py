"""
Tests for the in-process relay.

The relay exists so the protocol can be exercised end to end without a network,
and so the one thing a passive observer always sees -- how long each frame is --
can be measured. These tests cover both jobs, and treat the second one seriously:
a relay that reported a constant length for every frame would be worse than
useless, because it would make any padding scheme look good.
"""

import threading

import pytest

from src.p2p.relay import (
    DEFAULT_QUEUE_SIZE,
    Closed,
    Endpoint,
    FrameTooLarge,
    NoFrame,
    QueueFull,
    Relay,
    Transport,
    TransportError,
)
from src.protocol.session import accept, initiate
from src.protocol.wire import MAX_FRAME_BYTES
from tests.test_session import make_responder


def linked():
    relay = Relay()
    alice, bob = relay.link("alice", "bob")
    return relay, alice, bob


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_two_endpoints_see_each_other():
    _, alice, bob = linked()
    assert alice.peer == "bob"
    assert bob.peer == "alice"
    assert isinstance(alice, Transport)
    assert isinstance(alice, Endpoint)


def test_a_name_cannot_be_linked_twice():
    relay = Relay()
    relay.link("alice", "bob")
    with pytest.raises(TransportError):
        relay.link("alice", "carol")
    with pytest.raises(TransportError):
        relay.link("dave", "alice")


def test_an_endpoint_cannot_reach_itself():
    """
    A participant that could read its own inbound queue would be able to see
    what it sent, which no real network allows. The relay refuses to be built
    that way rather than leaving it to a caller to notice.
    """
    relay = Relay()
    with pytest.raises(TransportError):
        relay.link("alice", "alice")


def test_participants_are_listed_in_linking_order():
    relay = Relay()
    relay.link("alice", "bob")
    relay.link("carol", "dave")
    assert relay.participants() == ["alice", "bob", "carol", "dave"]


def test_a_relay_refuses_nonsense_configuration():
    with pytest.raises(ValueError):
        Relay(queue_size=0)
    with pytest.raises(ValueError):
        Relay(max_frame_bytes=0)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


def test_a_frame_arrives_whole_and_in_order():
    _, alice, bob = linked()
    for index in range(10):
        alice.send(b"frame %d" % index)
    for index in range(10):
        assert bob.receive(timeout=0.1) == b"frame %d" % index


def test_each_direction_is_separate():
    _, alice, bob = linked()
    alice.send(b"to bob")
    bob.send(b"to alice")
    assert bob.receive(timeout=0.1) == b"to bob"
    assert alice.receive(timeout=0.1) == b"to alice"


def test_receiving_with_nothing_queued_times_out():
    _, alice, _ = linked()
    with pytest.raises(NoFrame):
        alice.receive(timeout=0.05)


def test_a_full_queue_refuses_the_sender_instead_of_growing():
    """
    An unbounded buffer is a denial of service that needs no attacker: a peer
    that writes faster than the other side reads grows the queue until the
    process is gone. A refused frame leaves the receiver's session state intact,
    which is recoverable.
    """
    relay = Relay(queue_size=2)
    alice, bob = relay.link("alice", "bob")
    alice.send(b"one")
    alice.send(b"two")
    with pytest.raises(QueueFull):
        alice.send(b"three")
    # A frame the queue refused was never sent, so it is not in the observed
    # log either: the observation has to match what the peer received.
    assert relay.observed_lengths("alice") == [3, 3]
    # The two that were accepted are still there, in order.
    assert bob.receive(timeout=0.1) == b"one"
    assert bob.receive(timeout=0.1) == b"two"
    # And making room lets the next one through.
    alice.send(b"four")
    assert bob.receive(timeout=0.1) == b"four"
    assert relay.observed_lengths("alice") == [3, 3, 4]


def test_the_default_queue_is_bounded():
    assert 0 < DEFAULT_QUEUE_SIZE < 10**6


def test_an_oversized_frame_is_refused_before_it_is_queued():
    relay = Relay(max_frame_bytes=64)
    alice, bob = relay.link("alice", "bob")
    with pytest.raises(FrameTooLarge):
        alice.send(b"x" * 65)
    # Nothing was recorded and nothing is waiting.
    assert relay.observed == ()
    alice.send(b"x" * 64)
    assert bob.receive(timeout=0.1) == b"x" * 64


def test_the_default_frame_limit_is_the_wire_limit():
    _, alice, _ = linked()
    with pytest.raises(FrameTooLarge):
        alice.send(b"x" * (MAX_FRAME_BYTES + 1))


def test_a_frame_has_to_be_bytes():
    _, alice, _ = linked()
    with pytest.raises(TransportError):
        alice.send("a string, not a frame")


def test_a_bytearray_is_accepted_and_copied():
    """
    A bytearray is accepted because it is the obvious thing to have in hand, and
    copied because the caller could otherwise mutate a frame already in the
    queue -- which would mean the transport carried something the sender never
    encrypted.
    """
    _, alice, bob = linked()
    frame = bytearray(b"mutable")
    alice.send(frame)
    frame[0] = ord("M")
    assert bob.receive(timeout=0.1) == b"mutable"


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------


def test_a_closed_endpoint_refuses_both_directions():
    relay, alice, bob = linked()
    alice.send(b"queued before the close")
    alice.close()
    with pytest.raises(Closed):
        alice.send(b"after")
    with pytest.raises(Closed):
        alice.receive(timeout=0.05)
    # The peer cannot deliver to it either, so a half-closed link cannot be used
    # to keep a conversation going on one side.
    with pytest.raises(Closed):
        bob.send(b"to a closed endpoint")
    # Dropping an endpoint drops its queue with it, so the frame that was in
    # flight is gone. That is why the honest order is drain, then close, and why
    # a real transport that promises delivery has to say what it does with a
    # half-delivered stream. This one says it by dropping it.
    assert relay.is_closed("alice")
    assert "alice" not in relay.participants()


def test_an_endpoint_works_as_a_context_manager():
    relay = Relay()
    with relay.link("alice", "bob")[0] as alice:
        alice.send(b"in flight")
    assert relay.is_closed("alice")


def test_closing_the_relay_closes_everything():
    relay = Relay()
    alice, bob = relay.link("alice", "bob")
    relay.close()
    for endpoint in (alice, bob):
        with pytest.raises(Closed):
            endpoint.send(b"nope")
    assert relay.participants() == []


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def test_every_frame_is_recorded_with_its_length():
    relay, alice, bob = linked()
    assert relay.observed == ()
    alice.send(b"12345")
    bob.send(b"123")
    assert relay.observed == (("alice", 5), ("bob", 3))
    assert relay.observed_lengths() == [5, 3]
    assert relay.observed_lengths("alice") == [5]
    assert relay.observed_lengths("bob") == [3]


def test_the_observation_is_a_copy():
    relay, alice, _ = linked()
    alice.send(b"x")
    snapshot = relay.observed
    alice.send(b"yy")
    assert snapshot == (("alice", 1),)
    assert relay.observed == (("alice", 1), ("alice", 2))


def test_observation_can_be_reset_without_disturbing_the_queues():
    relay, alice, bob = linked()
    alice.send(b"before")
    relay.reset_observation()
    assert relay.observed == ()
    assert bob.receive(timeout=0.1) == b"before"
    alice.send(b"after")
    assert relay.observed_lengths() == [5]


def test_the_relay_reports_the_real_length_difference():
    """
    The measurement has to be capable of showing a difference.

    If two different messages produced the same recorded length here, a padding
    test built on this relay would pass for the wrong reason, and padding would
    be declared working on the strength of a broken instrument.
    """
    relay, alice, _ = linked()
    alice.send(b"x")
    alice.send(b"x" * 1000)
    assert relay.observed_lengths() == [1, 1000]


# ---------------------------------------------------------------------------
# A whole conversation over the relay
# ---------------------------------------------------------------------------


def test_a_session_runs_over_the_relay():
    relay, alice_endpoint, bob_endpoint = linked()
    responder, responder_public, _ = make_responder()
    from src.crypto.x3dh import X3DH

    initiator_private, initiator_public = X3DH().generate_identity_keys()

    initiator, init_frame = initiate(initiator_private, responder.publish_bundle())
    alice_endpoint.send(init_frame)

    responder_session, confirmation = accept(responder, bob_endpoint.receive())
    bob_endpoint.send(confirmation)
    initiator.confirm(alice_endpoint.receive())

    alice_endpoint.send(initiator.send(b"over the relay"))
    assert responder_session.receive(bob_endpoint.receive()) == b"over the relay"

    bob_endpoint.send(responder_session.send(b"and back"))
    assert initiator.receive(alice_endpoint.receive()) == b"and back"

    assert initiator.peer_identity == responder_public
    assert initiator.local_identity == initiator_public


def test_a_whole_conversation_under_concurrent_senders():
    """
    Two threads, one relay, one session.

    The relay is the only shared mutable object here, so this is a test of its
    locking as much as of the protocol. A frame that arrived interleaved or lost
    would show up as a decryption failure.
    """
    relay, alice_endpoint, bob_endpoint = linked()
    responder, _, _ = make_responder(prekeys=4)
    from src.crypto.x3dh import X3DH

    initiator_private, _ = X3DH().generate_identity_keys()

    initiator, init_frame = initiate(initiator_private, responder.publish_bundle())
    alice_endpoint.send(init_frame)
    responder_session, confirmation = accept(responder, bob_endpoint.receive())
    bob_endpoint.send(confirmation)
    initiator.confirm(alice_endpoint.receive())
    # The handshake frames are not what this test is about.
    relay.reset_observation()

    sent = []
    received = []
    errors = []

    def outbound(count, base):
        try:
            for index in range(count):
                payload = b"%s-%d" % (base, index)
                alice_endpoint.send(initiator.send(payload))
                sent.append(payload)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def inbound(count):
        try:
            for _ in range(count):
                received.append(responder_session.receive(bob_endpoint.receive()))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    total = 40
    producer = threading.Thread(target=outbound, args=(total, b"msg"))
    consumer = threading.Thread(target=inbound, args=(total,))
    producer.start()
    consumer.start()
    producer.join(timeout=10)
    consumer.join(timeout=10)

    assert errors == []
    assert received == sent
    # And every one of them was individually framed: the relay did not coalesce
    # two messages into one, which would have hidden the very thing it exists to
    # make observable.
    assert len(relay.observed_lengths("alice")) == total
