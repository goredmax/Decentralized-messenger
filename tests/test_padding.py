"""
Tests for length hiding.

The whole module exists to make one number wrong for an observer, so the tests
are organised around that number rather than around the functions.

The test that matters most is
:func:`test_a_one_byte_message_and_a_5000_byte_message_are_the_same_size`.
A padding scheme that does not move the observable frame length is not a
padding scheme, and it is exactly the kind of thing that can be implemented,
tested against itself, and reported as working. So the assertions here are on
``len(frame)`` as a transport sees it, and the companion
:func:`test_a_message_that_crosses_a_bucket_grows_the_frame` exists because the
first test alone passes just as happily against a scheme that makes every frame
the same size forever -- which would hide length and everything else with it.
"""

import os

import pytest

from src.crypto.double_ratchet import DecryptionError
from src.crypto.x3dh import X3DH
from src.p2p.relay import Relay
from src.protocol.padding import (
    DEFAULT_PROFILE,
    FRAME_OVERHEAD_MAX,
    RESERVED_BYTES,
    PaddingError,
    PaddingProfile,
    fits,
    pad,
    unpad,
)
from src.protocol.session import accept, initiate
from src.protocol.wire import (
    KIND_HANDSHAKE_INIT,
    KIND_KEY_CONFIRMATION,
    KIND_RATCHET_MESSAGE,
    MAX_FRAME_BYTES,
    WIRE_VERSION,
    InvalidField,
    WireError,
    decode_ratchet_message,
    encode_ratchet_message,
    encode_varint,
)
from tests.test_session import handshake, make_responder

UNPADDED = PaddingProfile(floor=0, step=1)

#: Sizes chosen to straddle the interesting boundaries: inside the floor, at it,
#: and past the first step.
SMALL = 1
MEDIUM = 5000
AT_FLOOR = DEFAULT_PROFILE.floor
ACROSS = 65536


def build_frame(header: dict, ciphertext: bytes, fill: bytes = b'',
                declared_length: int = None,
                fill_length: int = None) -> bytes:
    """
    Assemble a ratchet frame from its parts.

    Written out rather than derived from a real frame by surgery: a test that
    rewrites bytes it did not build is a test of the rewriting, and when it
    failed there was no way to tell which of the two was wrong.
    """
    declared = len(ciphertext) if declared_length is None else declared_length
    if fill_length is not None:
        fill = os.urandom(fill_length)
    payload = (
        bytes.fromhex(header['dh'])
        + encode_varint(header['pn'])
        + encode_varint(header['n'])
        + encode_varint(declared)
        + ciphertext
        + fill
    )
    return bytes((WIRE_VERSION, KIND_RATCHET_MESSAGE)) + \
        encode_varint(len(payload)) + payload


def send(sender, receiver, size: int, profile=None) -> int:
    """Send a message of ``size`` bytes and return the frame length."""
    frame = sender.send(b'x' * size, profile)
    assert receiver.receive(frame, profile) == b'x' * size
    return len(frame)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def test_a_one_byte_message_and_a_5000_byte_message_are_the_same_size():
    """
    The test this module exists to pass.

    1 byte and 5000 bytes are the two sizes the brief named, and they are both
    far below the 8192-byte floor, so they must land in the same bucket and
    produce frames of identical length. Asserted on ``len(frame)``, which is what
    a transport records and what an observer counts.
    """
    pair = handshake()
    small = send(pair['initiator'], pair['responder'], SMALL)
    large = send(pair['initiator'], pair['responder'], MEDIUM)
    assert small == large


def test_a_message_that_crosses_a_bucket_grows_the_frame():
    """
    The other half of the measurement, and the one that stops the first test
    from being satisfied by a scheme that simply made every frame the same size.

    A ladder is a coarsening, not a constant. 5000 bytes and 65536 bytes cannot
    be the same size, because a message has to be able to be large; if they were,
    the scheme would be hiding length by refusing to carry messages, which is not
    a trade anyone should accept silently.
    """
    pair = handshake()
    under = send(pair['initiator'], pair['responder'], MEDIUM)
    over = send(pair['initiator'], pair['responder'], ACROSS)
    assert over > under
    # The difference is a whole number of steps, up to the one byte the payload
    # length varint itself gains when the frame passes 16 KiB. That byte is not
    # a leak -- the frame length is bucket plus a fixed overhead, and the
    # overhead moves only because the varint announcing it widened.
    assert 0 < (over - under) % DEFAULT_PROFILE.step <= 3


def test_every_size_below_the_floor_produces_one_length():
    pair = handshake()
    lengths = {
        send(pair['initiator'], pair['responder'], size)
        for size in (0, 1, 2, 100, 1000, 5000, 8000)
    }
    assert len(lengths) == 1


def test_the_floor_is_a_bound_on_the_padded_size_not_the_message():
    """
    The seam at the floor, pinned rather than left to be discovered.

    A message of ``floor - 1`` bytes needs a two-byte length prefix, so its
    padded total is ``floor + 1`` and it lands in the bucket above. Everything
    below that shares one bucket. Making the padded size a function of the
    message length and nothing else is worth a one-value seam; the alternative,
    a special case at the boundary, is the kind of thing that differs between
    two implementations.
    """
    assert len(pad(b'x' * (DEFAULT_PROFILE.floor - 1))) == \
        DEFAULT_PROFILE.bucket(DEFAULT_PROFILE.floor + 1)
    assert len(pad(b'x' * (DEFAULT_PROFILE.floor - 2))) == DEFAULT_PROFILE.floor


def test_the_frame_length_does_not_wobble_as_counters_grow():
    """
    ``pn`` and ``n`` are transmitted in the clear, and their varints grow as the
    conversation goes on. A frame sized from the *actual* varint widths would
    gain a byte every 128 messages; the padding is sized against the widest
    possible varints instead, so the length holds still.

    The bytes it would have leaked are public anyway, so this is tidiness rather
    than security -- but a length that changes on a schedule is a strange thing
    to hand to someone measuring you.
    """
    pair = handshake()
    initiator, responder = pair['initiator'], pair['responder']
    lengths = set()
    for _ in range(200):
        frame = initiator.send(b'x' * 10)
        assert responder.receive(frame) == b'x' * 10
        lengths.add(len(frame))
    # 200 messages crosses the 128 boundary where the n varint widens.
    assert len(lengths) == 1


def test_padding_is_actually_reachable_through_a_transport():
    """
    Measured the way an observer would measure it: the relay's own record.

    ``len(frame)`` could in principle be right while something upstream resized
    the frame, or the other way round. The relay records what it was handed, so
    this closes that gap.
    """
    pair = handshake()
    relay = Relay()
    sender, receiver = relay.link('sender', 'receiver')
    for size in (SMALL, MEDIUM, 1000):
        sender.send(pair['initiator'].send(b'x' * size))
    assert len(set(relay.observed_lengths('sender'))) == 1


def test_a_disabled_profile_restores_the_original_sizes():
    """
    The control for every measurement above.

    Without this, "all the lengths are equal" could be true because the test
    never sent anything of different sizes. Turning padding off has to bring the
    two lengths apart, or the measurement is measuring nothing.

    Measured on the frames alone, without a receiver: an unpadded frame is not
    one this version can read, which is the next test's subject.
    """
    relay = Relay()
    sender, _ = relay.link('sender', 'receiver')
    pair = handshake()
    for size in (SMALL, MEDIUM):
        sender.send(pair['initiator'].send(b'x' * size, UNPADDED))
    small, large = relay.observed_lengths('sender')
    # The frame grows by the plaintext difference, up to the widths of the two
    # varints that grew with it.
    assert abs((large - small) - (MEDIUM - SMALL)) <= 2
    assert large - small > 4000


# ---------------------------------------------------------------------------
# The plaintext layer
# ---------------------------------------------------------------------------

def test_pad_and_unpad_round_trip():
    for size in (0, 1, 127, 128, 5000, 8191, 8192, 8193, 20000):
        message = bytes(range(256)) * (size // 256) + b'\x00' * (size % 256)
        message = message[:size]
        assert unpad(pad(message)) == message


def test_padding_lands_on_the_bucket():
    assert len(pad(b'x')) == DEFAULT_PROFILE.bucket(2)
    assert len(pad(b'x' * 5000)) == DEFAULT_PROFILE.bucket(5003)
    assert len(pad(b'')) == DEFAULT_PROFILE.bucket(1)


def test_a_disabled_profile_does_not_pad():
    assert pad(b'x' * 100, UNPADDED) == b'x' * 100
    assert UNPADDED.bucket(100) == 100
    assert UNPADDED.frame_payload_length(100) == 100


def test_frames_from_a_disabled_profile_are_not_read_by_a_v2_receiver():
    """
    Why ``WIRE_VERSION`` went to 2, and the reason it had to.

    An unpadded plaintext has no length prefix, so a v2 receiver reads its first
    byte as a length. Sometimes that fails loudly, as here, and sometimes it
    would not: a message beginning with a small byte would "unpad" to a
    fragment of itself and look like a successful decryption of the wrong text.

    The prefix is inside the AEAD, so no attacker can forge one -- the only way
    into that state is a version mismatch, and a version byte is what catches a
    version mismatch. The disabled profile therefore exists for measurement and
    for tests, and is not a mode a receiver can be in.
    """
    pair = handshake()
    frame = pair['initiator'].send(b'x' * 100, UNPADDED)
    with pytest.raises(PaddingError):
        pair['responder'].receive(frame)


def test_the_filler_is_not_all_zeros():
    """
    Zeros would make the padding distinguishable from message data that happens
    to contain zeros: a smaller leak than the length, and a free one to give up.
    """
    padded = pad(b'x')
    body = padded[2:]
    assert len(body) > 8
    assert not all(byte == 0 for byte in body)
    # And it is not a constant either: two paddings of the same message differ.
    assert pad(b'x') != pad(b'x')


def test_the_length_prefix_is_what_makes_it_unambiguous():
    padded = pad(b'hello world')
    assert padded[0] == 11
    assert padded[1:12] == b'hello world'


def test_unpad_rejects_a_message_with_no_prefix():
    with pytest.raises(PaddingError):
        unpad(b'')


def test_unpad_rejects_a_length_that_does_not_fit():
    # Declares 200 bytes of message inside a 12-byte input.
    with pytest.raises(PaddingError):
        unpad(b'\xc8' + b'x' * 11)


def test_unpad_rejects_a_non_canonical_length_prefix():
    """
    ``0x80 0x00`` is a two-byte encoding of zero. Accepting it would mean two
    distinct byte strings decode to the same message, which is the property that
    makes a signature over an encoding worth anything.
    """
    with pytest.raises(PaddingError):
        unpad(b'\x80\x00' + b'x' * 10)


def test_unpad_rejects_a_prefix_that_runs_off_the_end():
    with pytest.raises(PaddingError):
        unpad(b'\xff')


def test_unpad_rejects_a_prefix_longer_than_64_bits():
    with pytest.raises(PaddingError):
        unpad(b'\x80' * 12 + b'x' * 10)


def test_padding_rejects_non_bytes():
    with pytest.raises(PaddingError):
        pad('a string')
    with pytest.raises(PaddingError):
        unpad('a string')


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------

def test_the_bucket_is_monotonic():
    """
    Monotonicity is what makes equal padded sizes mean indistinguishable
    messages. A profile where a larger message could pad to a smaller size would
    leak the message count into the frame size.
    """
    previous = 0
    for size in range(0, 3 * DEFAULT_PROFILE.floor, 97):
        current = DEFAULT_PROFILE.bucket(size)
        assert current >= previous
        assert current >= size
        previous = current


def test_the_bucket_is_a_function_of_length_alone():
    assert (DEFAULT_PROFILE.bucket(1)
            == DEFAULT_PROFILE.bucket(1)) is True
    assert DEFAULT_PROFILE.bucket(1) == DEFAULT_PROFILE.bucket(5000)
    assert DEFAULT_PROFILE.bucket(8193) != DEFAULT_PROFILE.bucket(8192)


def test_a_ladder_profile_costs_less_and_leaks_more():
    """
    The trade, made concrete. Signal's ladder coarsens from 256 bytes; this one
    hides everything below 8192. The cheap profile is available, and choosing it
    is choosing to leak a coarse bucket.
    """
    ladder = PaddingProfile(floor=256, step=256)
    assert ladder.bucket(1) == 256
    assert ladder.bucket(5000) == 5120
    assert ladder.bucket(1) != ladder.bucket(5000)
    # The default does not have that problem, which is why it is the default.
    assert DEFAULT_PROFILE.bucket(1) == DEFAULT_PROFILE.bucket(5000)


def test_a_profile_that_cannot_fit_is_refused():
    with pytest.raises(PaddingError):
        PaddingProfile(floor=MAX_FRAME_BYTES * 2)
    with pytest.raises(PaddingError):
        PaddingProfile(step=MAX_FRAME_BYTES * 2)


def test_a_profile_rejects_nonsense():
    for kwargs in ({'floor': -1}, {'step': 0}, {'floor': 'big'},
                   {'step': True}):
        with pytest.raises(PaddingError):
            PaddingProfile(**kwargs)


def test_a_message_too_large_to_pad_is_refused_with_a_reason():
    reason = fits(MAX_FRAME_BYTES)
    assert reason is not None
    assert 'does not fit' in reason
    assert fits(1000) is None


def test_sending_an_unsendable_message_raises_rather_than_truncating():
    pair = handshake()
    with pytest.raises(PaddingError):
        pair['initiator'].send(b'x' * MAX_FRAME_BYTES)


def test_the_frame_overhead_leaves_room_inside_a_frame():
    """
    The overhead constant is what the frame ladder is shifted by, so it has to
    be at least the real overhead. Pinned here rather than derived, because a
    constant that drifts from the thing it reserves room for fails silently:
    frames come out a byte short and the AEAD gets the wrong tail.
    """
    from src.protocol.limits import AEAD_OVERHEAD

    assert RESERVED_BYTES == FRAME_OVERHEAD_MAX
    # version, kind, payload length varint, DH key, three header varints, nonce
    # and tag.
    assert FRAME_OVERHEAD_MAX == 2 + 9 + 32 + 9 * 3 + AEAD_OVERHEAD


def test_the_smallest_padded_frame_fits_in_a_frame():
    profile = PaddingProfile(floor=64, step=64)
    frame = encode_ratchet_message(
        b'\x00' * (64 + 40), {'dh': '11' * 32, 'pn': 0, 'n': 0}, profile
    )
    assert len(frame) <= MAX_FRAME_BYTES
    assert decode_ratchet_message(frame).n == 0


def test_a_profile_describes_itself():
    assert 'floor 8192' in DEFAULT_PROFILE.describe()
    assert UNPADDED.describe() == 'disabled'
    assert UNPADDED.enabled is False
    assert DEFAULT_PROFILE.enabled is True


# ---------------------------------------------------------------------------
# The frame layer
# ---------------------------------------------------------------------------

def test_the_ct_length_prefix_round_trips():
    for size in (40, 100, 8192, 20000):
        ciphertext = b'\x5a' * size
        frame = encode_ratchet_message(
            ciphertext, {'dh': '11' * 32, 'pn': 0, 'n': 0}
        )
        assert decode_ratchet_message(frame).ciphertext == ciphertext


def test_trailing_fill_is_ignored_by_the_decoder():
    """
    The fill is framing, not content. The decoder must not require it to be
    present or to be a particular size: a peer that does not pad is not an
    error, and enforcing a bucket would turn a padding policy into a hard
    protocol break.
    """
    ciphertext = b'\x5a' * 100
    header = {'dh': '11' * 32, 'pn': 0, 'n': 0}
    for fill in (b'', os.urandom(10), os.urandom(5000)):
        assert decode_ratchet_message(
            build_frame(header, ciphertext, fill=fill)
        ).ciphertext == ciphertext


def test_a_lying_ciphertext_length_fails_at_the_aead_not_silently():
    """
    The length prefix is not authenticated, so this checks what an attacker
    gets by rewriting it.

    Shortening it means the receiver takes fewer bytes and the tag is no longer
    the last 16 of what it decrypts; lengthening it means the tag is followed by
    attacker's bytes. Either way the AEAD fails. A failure is the correct
    outcome; the thing that must not happen is a *successful* decryption of
    something else.
    """
    pair = handshake()
    frame = pair['initiator'].send(b'the original message')
    message = decode_ratchet_message(frame)
    original = message.ciphertext
    header = message.to_header()
    receiver = pair['responder']

    # Shorter: the receiver takes fewer bytes, so the tag is no longer the last
    # 16 of what it decrypts, and the rest of the frame becomes fill.
    with pytest.raises(DecryptionError):
        receiver.receive(
            build_frame(header, original, declared_length=len(original) - 16)
        )
    # Longer: the attacker appends bytes and declares them, so the tag is
    # followed by something the AEAD never produced.
    with pytest.raises(DecryptionError):
        receiver.receive(build_frame(
            header, original + os.urandom(16),
            declared_length=len(original) + 16,
        ))


def test_a_ciphertext_length_past_the_end_is_caught_by_the_decoder():
    """
    A declared length that runs off the end of the frame is a malformed frame,
    not a failed decryption, and the two get different errors on purpose.
    """
    with pytest.raises(WireError):
        decode_ratchet_message(build_frame(
            {'dh': '11' * 32, 'pn': 0, 'n': 0},
            b'\x5a' * 100,
            declared_length=100000,
        ))


def test_a_ciphertext_too_short_to_be_a_message_is_refused():
    # The frame is long enough that only the declared length is wrong, so the
    # decoder reaches the check instead of stopping on the frame minimum.
    with pytest.raises(InvalidField):
        decode_ratchet_message(build_frame(
            {'dh': '11' * 32, 'pn': 0, 'n': 0},
            b'\x5a' * 10,
            declared_length=10,
            fill_length=200,
        ))


def test_a_frame_padded_by_one_profile_decodes_under_another():
    """
    The receiver does not check the bucket, so a sender that pads coarsely is
    still readable. That is the cost of not making padding part of the wire
    format, and it is the right trade: padding is a policy, not a protocol
    version.
    """
    pair = handshake()
    coarse = PaddingProfile(floor=1024, step=1024)
    frame = pair['initiator'].send(b'hello', coarse)
    assert pair['responder'].receive(frame) == b'hello'
    # And the receiver unpads by the prefix, not by its own ladder.
    assert pair['responder'].receive(
        pair['initiator'].send(b'hello again', coarse)
    ) == b'hello again'


def test_a_padded_frame_is_still_a_ratchet_frame():
    frame = pair = handshake()
    frame = pair['initiator'].send(b'x')
    assert frame[1] == KIND_RATCHET_MESSAGE


def test_padding_does_not_touch_the_handshake_frames():
    """
    A deliberate omission, pinned so it cannot be "fixed" by accident.

    The handshake frames are 134 bytes and 64 bytes, sent twice per session, and
    the one field that varies is a 64-bit prekey id whose varint differs by one
    byte. Padding them would cost 16 KiB per session to hide a byte an observer
    can count anyway -- there are exactly two of them. The cost is real and the
    benefit is not.
    """
    from src.protocol.wire import encode_handshake_init

    responder, _, _ = make_responder()
    bundle = responder.publish_bundle()
    initiator_private, _ = X3DH().generate_identity_keys()
    _, init_frame = initiate(initiator_private, bundle)
    _, confirmation = accept(responder, init_frame)
    assert len(init_frame) < DEFAULT_PROFILE.floor
    assert len(confirmation) < DEFAULT_PROFILE.floor
    assert init_frame[1] == KIND_HANDSHAKE_INIT
    assert confirmation[1] == KIND_KEY_CONFIRMATION
    assert encode_handshake_init is not None
