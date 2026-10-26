"""
Tests for the sender key transparency log.

The claim this module makes is narrow, and the tests are built around the four
things it actually detects rather than around the API:

- a **replay of the chain already trusted**, which is the case trust on first
  use structurally cannot see, because the replayed chain *is* the trusted one;
- a **fork**, where two members have been shown different histories;
- a **rollback**, where a log is truncated to a point the client has reached;
- a **forged entry**, which is not signed by the sender it names.

The fifth test is the one that keeps the rest honest: a log truncated at the tail
*is* accepted, because nothing in a one-sided log can tell "nothing new" from
"nothing more exists". A suite that only demonstrated the four detections would
read as a claim of full transparency, which is not what this is.
"""

import pytest
from nacl.signing import SigningKey

from src.protocol.sender_key import (
    AttestationStatus,
    InconsistentLog,
    RollbackDetected,
    SenderKeyId,
    SenderKeySessionBuilder,
    SenderKeyState,
)
from src.protocol.transparency import (
    GENESIS,
    Anchor,
    LogEntry,
    LogFormatError,
    LogLinkError,
    LogRollback,
    LogSignatureError,
    SenderKeyTransparency,
    TransparencyLog,
)

GROUP = "a-group"


def member():
    return SigningKey.generate()


def first_chain(identity, group_id=GROUP):
    return SenderKeyState.create(group_id, identity)


def distribution(state, identity):
    return state.distribution_message().sign(identity)


def populated(*pairs):
    """A log with one entry per (identity, state) pair, in the given order."""
    log = TransparencyLog()
    for identity, state in pairs:
        log.append_distribution(identity, distribution(state, identity))
    return log


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def test_an_entry_verifies_and_hashes():
    identity = member()
    state = first_chain(identity)
    entry = LogEntry(
        index=0,
        key_id=state.key_id,
        chain_id=state.chain_id,
        iteration=0,
        previous=GENESIS,
        signature=b"\x00" * 64,
    ).sign(identity)
    entry.verify()
    assert len(entry.hash()) == 32
    assert entry.hash() == entry.hash()


def test_an_entry_cannot_be_signed_by_anyone_but_its_sender():
    identity, other = member(), member()
    state = first_chain(identity)
    entry = LogEntry(
        index=0,
        key_id=state.key_id,
        chain_id=state.chain_id,
        iteration=0,
        previous=GENESIS,
        signature=b"\x00" * 64,
    )
    with pytest.raises(LogFormatError):
        entry.sign(other)


def test_a_forged_signature_is_caught():
    identity, other = member(), member()
    state = first_chain(identity)
    forged = LogEntry(
        index=0,
        key_id=state.key_id,
        chain_id=state.chain_id,
        iteration=0,
        previous=GENESIS,
        signature=other.sign(state.chain_id).signature,
    )
    with pytest.raises(LogSignatureError):
        forged.verify()


def test_changing_any_field_breaks_the_signature():
    """
    Every field is covered, one at a time.

    A field left out of the signed bytes would let a storage party rewrite it
    and keep the signature, which is the same class of defect as an unsigned
    field in a handshake transcript.
    """
    from dataclasses import replace

    identity = member()
    state = first_chain(identity)
    entry = LogEntry(
        index=3,
        key_id=state.key_id,
        chain_id=state.chain_id,
        iteration=7,
        previous=b"\x11" * 32,
        signature=b"\x00" * 64,
    ).sign(identity)
    entry.verify()

    for field, value in (
        ("index", 4),
        ("chain_id", b"\x22" * 32),
        ("iteration", 8),
        ("previous", b"\x33" * 32),
    ):
        with pytest.raises(LogSignatureError):
            replace(entry, **{field: value}).verify()

    other_group = replace(entry, key_id=SenderKeyId(GROUP + "!", state.key_id.sender))
    with pytest.raises(LogSignatureError):
        other_group.verify()

    other_sender = replace(entry, key_id=SenderKeyId(GROUP, member().verify_key))
    with pytest.raises(LogSignatureError):
        other_sender.verify()


def test_an_entry_refuses_nonsense_fields():
    key_id = first_chain(member()).key_id
    good = {
        "index": 0,
        "key_id": key_id,
        "chain_id": b"\x00" * 32,
        "iteration": 0,
        "previous": GENESIS,
        "signature": b"\x00" * 64,
    }
    for field, value in (
        ("index", -1),
        ("index", True),
        ("chain_id", b"\x00" * 31),
        ("iteration", -1),
        ("previous", b""),
        ("signature", b"\x00" * 63),
    ):
        with pytest.raises(LogFormatError):
            LogEntry(**dict(good, **{field: value}))


def test_an_entry_is_immutable():
    """
    Frozen, because an entry is a statement about a moment.

    A mutable one could be edited in place and its hash -- which is what the next
    entry commits to -- would no longer match what it was when that commitment
    was made.
    """
    identity = member()
    entry = LogEntry(
        index=0,
        key_id=first_chain(identity).key_id,
        chain_id=b"\x00" * 32,
        iteration=0,
        previous=GENESIS,
        signature=b"\x00" * 64,
    ).sign(identity)
    with pytest.raises(Exception):
        entry.index = 5


# ---------------------------------------------------------------------------
# The log itself
# ---------------------------------------------------------------------------


def test_a_log_links_and_verifies():
    identities = [member() for _ in range(3)]
    log = populated(*[(i, first_chain(i)) for i in identities])
    log.verify()
    assert log.size() == 3
    assert log.at(0).previous == GENESIS
    assert log.at(1).previous == log.at(0).hash()
    assert log.at(2).previous == log.at(1).hash()


def test_an_empty_log_links_to_genesis():
    log = TransparencyLog()
    assert log.head() is None
    assert log.head_hash() == GENESIS
    log.verify()


def test_a_log_refuses_a_distribution_that_does_not_verify():
    identity = member()
    state = first_chain(identity)
    unsigned = state.distribution_message()
    log = TransparencyLog()
    with pytest.raises(Exception):
        log.append_distribution(identity, unsigned)


def test_the_latest_chain_for_a_sender_is_found():
    identity = member()
    state = first_chain(identity)
    rotated = state.rotate()
    log = populated((identity, state), (identity, rotated))
    latest = log.latest_for(state.key_id)
    assert latest.index == 1
    assert latest.chain_id == rotated.chain_id
    assert len(log.chains_for(state.key_id)) == 2


def test_an_unknown_sender_has_no_latest():
    identity, stranger = member(), member()
    log = populated((identity, first_chain(identity)))
    assert log.latest_for(first_chain(stranger).key_id) is None


def test_two_senders_in_different_groups_do_not_collide():
    """
    A member joining a second group starts a fresh chain there.

    That must not look like a rotation, which is why the key is the pair and not
    the sender alone. The log has to agree with that.
    """
    identity = member()
    here = first_chain(identity, "group-one")
    there = first_chain(identity, "group-two")
    log = populated((identity, here), (identity, there))
    log.verify()
    assert log.latest_for(here.key_id).index == 0
    assert log.latest_for(there.key_id).index == 1
    assert len(log.chains_for(here.key_id)) == 1


def test_a_rewritten_entry_is_caught_by_verify():
    """
    A storage party that rewrites an entry and keeps the old signature.

    Built through :meth:`TransparencyLog.from_entries`, which is the path a
    client takes when a server hands it entries -- a test that edited a copy and
    then verified the original would have passed without checking anything.
    """
    from dataclasses import replace

    identity = member()
    state = first_chain(identity)
    log = populated((identity, state), (identity, state.rotate()))
    tampered = replace(log.at(1), previous=b"\x00" * 32)

    with pytest.raises(LogSignatureError):
        TransparencyLog.from_entries([log.at(0), tampered])


def test_a_dropped_entry_is_caught_by_from_entries():
    from dataclasses import replace

    identity = member()
    first, second, third = (
        first_chain(identity),
        first_chain(identity),
        first_chain(identity),
    )
    log = populated((identity, first), (identity, second), (identity, third))
    # Entry 2 with entry 1's link: the index follows, the link does not.
    spliced = replace(log.at(2), previous=log.at(0).hash())
    with pytest.raises(LogLinkError):
        TransparencyLog.from_entries([log.at(0), spliced])


def test_from_entries_accepts_an_honest_log():
    identity = member()
    state = first_chain(identity)
    log = populated((identity, state), (identity, state.rotate()))
    rebuilt = TransparencyLog.from_entries(log.entries())
    rebuilt.verify()
    assert rebuilt.size() == log.size()
    assert rebuilt.head_hash() == log.head_hash()


def test_entries_are_a_copy():
    identity = member()
    log = populated((identity, first_chain(identity)))
    entries = log.entries()
    log.append_distribution(identity, distribution(first_chain(identity), identity))
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# What the client sees
# ---------------------------------------------------------------------------


def test_a_client_adopts_a_log_and_anchors_it():
    identity = member()
    log = populated((identity, first_chain(identity)))
    view = SenderKeyTransparency()
    view.adopt(log)
    assert view.size() == 1
    assert view.anchor().index == 0
    assert view.anchor().entry_hash == log.at(0).hash()


def test_an_unknown_chain_is_unattested():
    identity = member()
    log = populated((identity, first_chain(identity)))
    view = SenderKeyTransparency()
    view.adopt(log)
    # A different member entirely: the log has never seen this key id, so it has
    # no opinion rather than a negative one.
    stranger = member()
    stranger_state = first_chain(stranger)
    assert (
        view.check(distribution(stranger_state, stranger))
        == AttestationStatus.UNATTESTED
    )


def test_the_latest_chain_is_current():
    identity = member()
    state = first_chain(identity)
    log = populated((identity, state))
    view = SenderKeyTransparency()
    view.adopt(log)
    assert view.check(distribution(state, identity)) == AttestationStatus.CURRENT


def test_an_older_chain_is_stale():
    """
    The property trust on first use cannot have.

    The chain is one the member has already accepted, so first-contact logic
    recognises it and moves on. Only a log knows that a newer chain exists.
    """
    identity = member()
    original = first_chain(identity)
    rotated = original.rotate()
    log = populated((identity, original), (identity, rotated))
    view = SenderKeyTransparency()
    view.adopt(log)
    assert view.check(distribution(original, identity)) == AttestationStatus.STALE
    assert view.check(distribution(rotated, identity)) == AttestationStatus.CURRENT


def test_a_gap_is_caught():
    """
    Entries removed from the middle.

    The one after the hole commits to a hash the client has not got, so it does
    not link and the batch is refused whole.
    """
    identity = member()
    log = populated(
        (identity, first_chain(identity)),
        (identity, first_chain(identity)),
        (identity, first_chain(identity)),
    )
    view = SenderKeyTransparency()
    with pytest.raises(LogLinkError):
        view.incorporate([log.at(0), log.at(2)])
    # Nothing was taken.
    assert view.size() == 0


def test_a_fork_is_caught():
    """
    Two members shown different histories.

    The divergent position is the first one where the hashes differ, and it is
    the same answer for anyone who asks both.
    """
    identity = member()
    # The same first two states in both logs, so the only divergence is at the
    # third. Two independently generated chains would differ everywhere and the
    # fork point would be meaningless.
    first, second, third = (
        first_chain(identity),
        first_chain(identity),
        first_chain(identity),
    )
    real = populated((identity, first), (identity, second), (identity, third))
    forked = populated(
        (identity, first), (identity, second), (identity, third.rotate())
    )

    assert real.fork_points(forked) == [2]
    assert forked.fork_points(real) == [2]

    view = SenderKeyTransparency()
    view.adopt(real)
    with pytest.raises(LogRollback):
        view.adopt(forked)


def test_a_rollback_to_an_earlier_point_is_caught():
    """
    A log truncated to a position the client has already reached.

    The case that an entry-wise check misses entirely: the truncated log contains
    nothing new, so every entry in it is one already held, and the rollback is
    invisible unless the log itself is compared against the anchor.
    """
    identity = member()
    first, second, third = (
        first_chain(identity),
        first_chain(identity),
        first_chain(identity),
    )
    full = populated((identity, first), (identity, second), (identity, third))
    short = populated((identity, first))
    assert full.entries()[:1] == short.entries()

    view = SenderKeyTransparency()
    view.adopt(full)
    assert view.anchor().index == 2
    with pytest.raises(LogRollback):
        view.adopt(short)


def test_a_forged_entry_is_caught():
    identity, impostor = member(), member()
    state = first_chain(identity)
    log = populated((identity, state))
    forged = LogEntry(
        index=1,
        key_id=state.key_id,
        chain_id=state.rotate().chain_id,
        iteration=0,
        previous=log.at(0).hash(),
        signature=impostor.sign(b"anything").signature,
    )
    view = SenderKeyTransparency()
    view.adopt(log)
    with pytest.raises(LogSignatureError):
        view.incorporate([forged])
    assert view.size() == 1


def test_a_rewritten_position_is_caught():
    """
    A *validly signed* entry at a position already held, with different content.

    Two defences are in the way on purpose, and this exercises the third. The
    signature fails if the entry is edited, so the attacker has to be the sender
    signing two different entries at one position -- which means a fork, which
    :meth:`TransparencyLog.verify` catches against the anchor before anything is
    incorporated. The check here is the one underneath, for a caller that
    incorporates entries one at a time without a log in hand.
    """
    identity = member()
    first, second = first_chain(identity), first_chain(identity)
    real = populated((identity, first), (identity, second))
    forked = populated((identity, first), (identity, second.rotate()))

    view = SenderKeyTransparency()
    view.adopt(real)
    divergent = forked.at(1)
    divergent.verify()  # genuinely signed; it is simply not what is held
    with pytest.raises(LogRollback):
        view.incorporate([divergent])


def test_a_tail_truncation_is_not_detected():
    """
    The honest limit, asserted rather than left to be discovered.

    Nothing has been shown since the last entry, and nothing says anything more
    exists. A one-sided log cannot tell that from the truth, and no amount of
    checking inside this module changes it -- the remedy is a second, independent
    source for the log's head, which is what a witness would be.
    """
    identity = member()
    state = first_chain(identity)
    log = populated((identity, state))
    view = SenderKeyTransparency()
    view.adopt(log)

    # The log gains an entry the client is not shown.
    later = state.rotate()
    log.append_distribution(identity, distribution(later, identity))
    assert log.size() == 2

    # Being re-shown what it already has changes nothing, and the view says
    # nothing about the entry it was not given.
    truncated = TransparencyLog.from_entries([log.at(0)])
    view.adopt(truncated)
    assert view.size() == 1
    assert view.anchor().index == 0
    # And it has no opinion about the chain it was not shown.
    assert view.check(distribution(later, identity)) == AttestationStatus.UNATTESTED


def test_a_fresh_client_can_be_shown_any_first_chain():
    """
    The other honest limit.

    With no anchor there is nothing to compare against, so a client's first
    contact with a sender is still trust on first use. The log raises the cost of
    getting it wrong on *subsequent* contacts, not the first one.
    """
    identity = member()
    state = first_chain(identity)
    log = populated((identity, state))
    view = SenderKeyTransparency()
    view.adopt(log)
    assert view.check(distribution(state, identity)) == AttestationStatus.CURRENT


def test_a_genesis_anchor_accepts_anything_and_forgets_nothing():
    identity = member()
    assert Anchor.genesis().index == -1
    view = SenderKeyTransparency(Anchor.genesis())
    view.adopt(populated((identity, first_chain(identity))))
    assert view.anchor().index == 0


def test_an_anchor_refuses_nonsense():
    with pytest.raises(LogFormatError):
        Anchor(index=-2, entry_hash=GENESIS)
    with pytest.raises(LogFormatError):
        Anchor(index=0, entry_hash=b"short")


def test_known_chains_are_reported_in_log_order():
    identity = member()
    first = first_chain(identity)
    second, third = first.rotate(), first_chain(identity).rotate()
    log = populated((identity, first), (identity, second), (identity, third))
    view = SenderKeyTransparency()
    view.adopt(log)
    assert view.known_chains(first.key_id) == [
        first.chain_id,
        second.chain_id,
        third.chain_id,
    ]


def test_a_client_resumes_from_a_stored_anchor():
    """
    The anchor's only purpose is surviving a restart, so it has to be
    constructible from a stored index and hash alone.
    """
    identity = member()
    state = first_chain(identity)
    log = populated((identity, state), (identity, state.rotate()))
    first_run = SenderKeyTransparency()
    first_run.adopt(log)
    stored = first_run.anchor()

    resumed = SenderKeyTransparency(stored)
    assert resumed.anchor() == stored
    log.append_distribution(identity, distribution(state.rotate(), identity))
    resumed.adopt(log)
    assert resumed.anchor().index == 2


# ---------------------------------------------------------------------------
# Through the session builder
# ---------------------------------------------------------------------------


def test_a_replayed_trusted_chain_is_caught_with_a_log():
    """
    The payoff, and the reason the log exists.

    Plain trust on first use accepts the replay: the chain in hand *is* the chain
    it trusts, so there is nothing to object to. With a log, the newer chain is
    on record and the replay is refused.
    """
    identity = member()
    original = first_chain(identity)
    rotated = original.rotate()
    log = populated((identity, original), (identity, rotated))

    plain = SenderKeySessionBuilder()
    assert plain.process(distribution(original, identity))[1] is False
    assert plain.process(distribution(original, identity))[1] is False

    builder = SenderKeySessionBuilder()
    view = SenderKeyTransparency()
    view.adopt(log)
    builder.process(distribution(rotated, identity), attestation=view)
    with pytest.raises(RollbackDetected):
        builder.process(distribution(original, identity), attestation=view)


def test_a_log_attested_rotation_needs_no_out_of_band_confirmation():
    """
    The other payoff: a rotation that the log vouches for is adopted.

    Without a log, every rotation is a question for a human, which is a social
    process, and social processes fail at inconvenient times.
    """
    identity = member()
    original = first_chain(identity)
    rotated = original.rotate()
    log = populated((identity, original), (identity, rotated))
    view = SenderKeyTransparency()
    view.adopt(log)

    plain = SenderKeySessionBuilder()
    plain.process(distribution(original, identity))
    assert plain.process(distribution(rotated, identity))[1] is True

    builder = SenderKeySessionBuilder()
    key_id, untrusted = builder.process(
        distribution(rotated, identity), attestation=view
    )
    assert untrusted is False
    assert builder.trusted_chains(key_id) == [rotated.chain_id]


def test_a_rotation_with_no_log_entry_still_needs_confirmation():
    """
    The log only speaks about chains it has seen.

    An unattested rotation is exactly as undecided as it was before the log
    existed, which is the right answer and not a disappointment: the log is
    evidence, not a rubber stamp.
    """
    identity = member()
    original = first_chain(identity)
    rotated = original.rotate()
    log = populated((identity, original))

    builder = SenderKeySessionBuilder()
    view = SenderKeyTransparency()
    view.adopt(log)
    builder.process(distribution(original, identity), attestation=view)
    assert builder.process(distribution(rotated, identity), attestation=view)[1] is True


def test_an_unattested_replay_of_the_trusted_chain_is_reported_not_raised():
    """
    The awkward middle case, decided explicitly.

    The chain in hand is the one already trusted, so refusing it would break a
    conversation the log has simply not caught up with yet. It is reported
    instead, so the caller can see that the log does not know about it.
    """
    identity = member()
    original = first_chain(identity)
    log = TransparencyLog()
    view = SenderKeyTransparency()

    builder = SenderKeySessionBuilder()
    builder.process(distribution(original, identity))
    key_id, untrusted = builder.process(
        distribution(original, identity), attestation=view
    )
    assert untrusted is True
    assert key_id == original.key_id
    assert not log.size()


def test_a_broken_log_is_refused_rather_than_ignored():
    """
    A log that does not hold together must not be treated as "no opinion".

    Reporting ``UNATTESTED`` for a log that has been tampered with would let the
    tampered case through the same path as a genuinely new chain, which is the
    one thing an attestation exists to prevent.
    """
    identity = member()
    original = first_chain(identity)
    rotated = original.rotate()

    class Broken:
        """An attestation that cannot be evaluated, and says so."""

        def check(self, message):
            return AttestationStatus.INCONSISTENT

    builder = SenderKeySessionBuilder()
    builder.process(distribution(original, identity))
    with pytest.raises(InconsistentLog):
        builder.process(distribution(rotated, identity), attestation=Broken())


def test_a_first_contact_is_accepted_even_if_the_log_has_another_sender():
    """
    Nothing to compare against, so nothing to object to.

    Refusing every first contact would make the protocol unusable, and a log that
    has entries for one sender says nothing at all about a different one.
    """
    known = member()
    stranger = member()
    log = populated(
        (known, first_chain(known)),
        (known, first_chain(known).rotate()),
    )
    view = SenderKeyTransparency()
    view.adopt(log)

    builder = SenderKeySessionBuilder()
    key_id, untrusted = builder.process(
        distribution(first_chain(stranger), stranger), attestation=view
    )
    assert untrusted is False
    assert key_id.sender == stranger.verify_key


def test_the_log_survives_a_group_scoped_rotation():
    """
    A member in two groups rotates in one of them.

    The key is the pair, so a rotation in one group is not a rotation in the
    other, and the log keeps them apart.
    """
    identity = member()
    here = first_chain(identity, "group-one")
    there = first_chain(identity, "group-two")
    rotated_here = here.rotate()
    log = populated((identity, here), (identity, there))
    log.append_distribution(identity, distribution(rotated_here, identity))
    view = SenderKeyTransparency()
    view.adopt(log)

    builder = SenderKeySessionBuilder()
    builder.process(distribution(there, identity), attestation=view)
    # The chain in group two is still the current one for group two.
    assert view.check(distribution(there, identity)) == AttestationStatus.CURRENT
    # And the one in group one is not.
    assert view.check(distribution(here, identity)) == AttestationStatus.STALE
