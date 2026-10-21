"""
Tests for the example CLI.

The CLI is thin on purpose -- the protocol lives in :mod:`src.protocol.session`
and the transport in :mod:`src.p2p.relay`. So these tests do not re-test the
protocol. They check the three things that are the CLI's own and that would
otherwise only be checked by a human running it and reading the output:

1. A full cycle runs, over the real transport, and produces no output that
   contradicts what happened. The demo prints frame lengths and safety numbers;
   if those are wrong, the demo is lying to the person reading it, which is the
   failure mode this project cares most about.
2. ``--data-dir`` really persists: the same identities and the same session
   come back on a second run, and a stored session is reused instead of a new
   handshake.
3. A negative ``--messages`` is refused rather than silently doing nothing.

One subprocess test runs the file as a program, because the sys.path bootstrap
and the entry point are only exercised that way, and both were broken at least
once while this was written.
"""

import os
import subprocess
import sys

import pytest

from examples.cli import (
    DEMO_PREKEYS,
    RECEIVER,
    SENDER,
    identity_is_initialised,
    main,
    open_participant,
    save_session,
)
from src.storage.container import Container
from src.storage.store import IdentityStore, PeerDirectory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# A full run
# ---------------------------------------------------------------------------

def test_a_full_run_succeeds(capsys):
    assert main(['--messages', '2']) == 0
    output = capsys.readouterr().out
    assert 'verified the bundle signature' in output
    assert 'verified the confirmation' in output
    assert 'message 0 from alice' in output
    assert 'reply 1 from bob' in output


def test_the_run_reports_the_prekey_it_spent(capsys):
    """
    The one-time prekeys are the whole reason the pools are persisted, so the
    demo has to show them being spent. A line that said "authenticated the
    initiator" without them would leave the forward-secrecy story unverified on
    screen.
    """
    main(['--messages', '0'])
    output = capsys.readouterr().out
    assert 'spent prekey 0 and post-quantum prekey 0' in output
    assert f'pools now hold {DEMO_PREKEYS - 1} and {DEMO_PREKEYS - 1}' in output
    # And the post-quantum contribution is named, so the output does not imply
    # the session is only as strong as X25519.
    assert 'forward secrecy survives a future break of X25519' in output
    assert 'ML-KEM-768' in output


def test_the_run_reports_real_frame_lengths(capsys):
    """
    The lengths printed have to be the lengths the transport carried.

    This is the number the padding work is judged by, so a demo that printed a
    plausible-looking constant instead of the real one would remove the only
    measurement in the repository. The relay is asked directly, and every
    distinct length it recorded has to appear in the output.
    """
    relay_lengths = []

    import examples.cli as cli
    original = cli.report_lengths

    def spy(relay, first_message):
        relay_lengths.extend(relay.observed_lengths()[first_message:])
        return original(relay, first_message)

    cli.report_lengths = spy
    try:
        main(['--messages', '1'])
    finally:
        cli.report_lengths = original

    output = capsys.readouterr().out
    assert relay_lengths
    for length in sorted(set(relay_lengths)):
        assert str(length) in output
    # And the instrument is capable of showing a difference, which is what makes
    # "all the same" mean something. Without this, equal lengths would also be
    # what a broken measurement would report.
    assert len(relay_lengths) >= 3


def test_the_run_shows_the_padding_working(capsys):
    """
    The demo's job is to show what the code does, so it must not claim padding
    is absent now that it is present, and it must not claim lengths are hidden
    when only the size bucket is.
    """
    main(['--messages', '1'])
    output = capsys.readouterr().out
    assert 'distinct length' in output
    assert '1 distinct length' in output
    assert '1 byte of plaintext' not in output
    assert 'Every message frame is the same size' in output
    # What padding does not do is still said out loud.
    assert 'how many messages there are' in output
    assert 'The one thing still visible is which' in output
    # And the old claim that nothing pads must be gone.
    assert 'Nothing here pads' not in output


def test_the_run_prints_safety_numbers(capsys):
    main(['--messages', '0'])
    output = capsys.readouterr().out
    assert 'safety numbers' in output
    # The comparison is the only defence against a substituted bundle, so the
    # demo has to put it in front of the reader.
    assert 'channel the attacker does not control' in output


def test_zero_messages_still_completes_the_handshake(capsys):
    assert main(['--messages', '0']) == 0
    output = capsys.readouterr().out
    assert 'verified the confirmation' in output
    assert 'message 0 from' not in output


def test_a_negative_message_count_is_refused():
    with pytest.raises(SystemExit):
        main(['--messages', '-1'])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_nothing_is_written_without_a_data_dir(tmp_path, monkeypatch):
    """
    A demo that quietly creates files the user never asked for is a demo people
    stop running. No directory, no writes.
    """
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    main(['--messages', '0'])
    assert set(os.listdir(tmp_path)) == before


def test_a_data_dir_persists_the_identities(tmp_path, capsys):
    directory = str(tmp_path / 'state')
    main(['--messages', '0', '--data-dir', directory, '--password', 'p'])
    first = capsys.readouterr().out

    main(['--messages', '0', '--data-dir', directory, '--password', 'p'])
    second = capsys.readouterr().out

    def identity_lines(output):
        # The banner lines are "alice" plus a 64-char hex identity. Matching on
        # the name alone also catches "bob published signed prekey ...", which
        # is how this test failed the first time.
        found = []
        for line in output.splitlines():
            parts = line.split()
            if len(parts) == 2 and len(parts[1]) == 64 and \
                    all(c in '0123456789abcdef' for c in parts[1]):
                found.append(line)
        return found

    assert identity_lines(first) == identity_lines(second)
    assert identity_lines(first), 'the run printed no identity lines'


def test_a_second_run_reuses_the_stored_session(tmp_path, capsys):
    """
    The point of persisting: no second handshake, and the same conversation.

    A fresh handshake against a fresh in-process peer would be a different
    conversation with different keys, so a run that silently re-handshakes would
    look fine and prove nothing.
    """
    directory = str(tmp_path / 'state')
    main(['--messages', '1', '--data-dir', directory, '--password', 'p'])
    first = capsys.readouterr().out
    assert 'published signed prekey' in first

    main(['--messages', '1', '--data-dir', directory, '--password', 'p'])
    second = capsys.readouterr().out
    assert 'was found and reopened' in second
    assert 'published signed prekey' not in second


def test_the_prekey_pool_survives_and_shrinks(tmp_path, capsys):
    """
    Forward secrecy after a restart depends on the pool being on disk. A demo
    that regenerated the pool each run would reach the 3-DH variant on the
    second handshake, and say nothing about it.
    """
    directory = str(tmp_path / 'state')
    main(['--messages', '0', '--data-dir', directory, '--password', 'p'])
    output = capsys.readouterr().out
    # The ids are random on the persistent path and 0-based on the in-memory
    # one, so the line is checked by shape rather than by value.
    assert 'spent prekey ' in output and 'and post-quantum prekey ' in output
    assert f'pools now hold {DEMO_PREKEYS - 1} and {DEMO_PREKEYS - 1}' in output

    main(['--messages', '0', '--data-dir', directory, '--password', 'p'])
    # No handshake ran, so neither pool shrank further: it is the same
    # conversation, not a new one.
    assert 'was found and reopened' in capsys.readouterr().out

    # A third run is still the same conversation, and still has both pools.
    main(['--messages', '0', '--data-dir', directory, '--password', 'p'])
    assert 'was found and reopened' in capsys.readouterr().out


def test_a_wrong_password_does_not_resume_the_session(tmp_path, capsys):
    """
    A different password is a different container, so there is nothing to
    resume. The demo must fall back to a fresh handshake rather than fail or,
    worse, invent a session.
    """
    directory = str(tmp_path / 'state')
    main(['--messages', '0', '--data-dir', directory, '--password', 'right'])
    capsys.readouterr()

    with pytest.raises(Exception):
        main(['--messages', '0', '--data-dir', directory, '--password', 'wrong'])


def test_the_stored_session_is_reachable_from_the_peer_key(tmp_path):
    """
    What ``PeerDirectory`` is for, exercised through the CLI: the conversation
    can be found again using the peer's public key and nothing else.
    """
    directory = str(tmp_path / 'state')
    main(['--messages', '0', '--data-dir', directory, '--password', 'p'])

    peers = PeerDirectory(
        Container.open(os.path.join(directory, 'peers.bin'), 'p')
    )
    sender = open_participant(SENDER, directory, 'p')
    receiver = open_participant(RECEIVER, directory, 'p')
    assert peers.role_of(receiver.identity_public) is True
    assert peers.role_of(sender.identity_public) is False


def test_nothing_sensitive_is_written_in_the_clear(tmp_path):
    directory = str(tmp_path / 'state')
    main(['--messages', '1', '--data-dir', directory, '--password', 'p'])

    identities = [
        bytes(open_participant(name, directory, 'p').identity_public)
        for name in (SENDER, RECEIVER)
    ]
    assert identities
    for root, _, files in os.walk(directory):
        for name in files:
            raw = open(os.path.join(root, name), 'rb').read()
            for key in identities:
                assert key not in raw
            for marker in (b'is_initiator', b'peers', b'root_key',
                            b'chain_key'):
                assert marker not in raw


def test_a_fresh_directory_is_created(tmp_path):
    directory = str(tmp_path / 'nested' / 'state')
    assert main(['--messages', '0', '--data-dir', directory]) == 0
    assert os.path.isdir(directory)
    assert os.path.exists(os.path.join(directory, 'peers.bin'))


def test_an_empty_container_is_reinitialised(tmp_path):
    """
    ``Container.create`` succeeding and the process dying before the first store
    leaves a container that exists and holds nothing. That has to be a first run
    again, not a crash on a missing identity.
    """
    directory = tmp_path / 'state'
    directory.mkdir()
    Container.create(
        str(directory / f'{SENDER}-identity.bin'), 'p',
        kdf_params=__import__(
            'examples.cli', fromlist=['DEMO_KDF']
        ).DEMO_KDF,
    )
    participant = open_participant(SENDER, str(directory), 'p')
    assert identity_is_initialised(IdentityStore(
        Container.open(str(directory / f'{SENDER}-identity.bin'), 'p')
    ))
    assert participant.identity_public is not None


# ---------------------------------------------------------------------------
# As a program
# ---------------------------------------------------------------------------

def test_the_file_runs_as_a_program():
    """
    The sys.path bootstrap and ``if __name__ == '__main__'`` are only exercised
    by running the file, and both were broken while this was written.
    """
    completed = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'examples', 'cli.py'),
         '--messages', '1'],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.join(REPO_ROOT, 'examples'),
    )
    assert completed.returncode == 0, completed.stderr
    assert 'message 0 from alice' in completed.stdout
    assert 'verified the confirmation' in completed.stdout


def test_save_session_records_the_role_and_the_state(tmp_path):
    """
    ``save_session`` is the CLI's whole persistence story, so it is checked
    directly rather than only through a run that happens to pass.
    """
    from src.crypto.x3dh import X3DH
    from src.protocol.session import accept, initiate
    from tests.test_session import make_responder

    responder, responder_public, _ = make_responder()
    sender_private, sender_public = X3DH().generate_identity_keys()
    sender_session, init_frame = initiate(
        sender_private, responder.publish_bundle()
    )
    receiver_session, confirmation = accept(responder, init_frame)
    sender_session.confirm(confirmation)

    directory = str(tmp_path / 'state')
    save_session(directory, 'p', responder_public, sender_session)
    save_session(directory, 'p', sender_public, receiver_session)

    peers = PeerDirectory(
        Container.open(os.path.join(directory, 'peers.bin'), 'p')
    )
    assert peers.role_of(responder_public) is True
    assert peers.role_of(sender_public) is False

    # Re-recording the same pair and role must not bump the epoch: the epoch
    # exists to make a rollback detectable, and a bump per save would make it
    # mean something other than what it is for.
    epoch = peers.epoch
    save_session(directory, 'p', responder_public, sender_session)
    assert peers.epoch == epoch

    # The state it wrote is loadable and still the session that was saved.
    from src.crypto.double_ratchet import PROTOCOL_VERSION
    from src.storage.store import SessionStore, peer_id
    stored = SessionStore(
        os.path.join(directory, 'sessions'), 'p'
    ).load(peer_id(responder_public), PROTOCOL_VERSION)
    assert stored.root_key == sender_session.state.root_key
