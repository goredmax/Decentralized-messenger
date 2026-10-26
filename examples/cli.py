"""
Two roles, one process, the whole cycle: bundle, X3DH, ratchet, messages.

    python examples/cli.py
    python examples/cli.py --messages 3
    python examples/cli.py --data-dir ./state

What this proves, and what it does not
--------------------------------------
It proves the protocol joins up. Before this existed, no code in the repository
constructed a Double Ratchet, and a conversation could only be assembled by hand
out of unrelated pieces; here the frames go out over a
:class:`~src.p2p.relay.Transport`, come back in, and decrypt.

It does not make a messenger. Both roles live in one process, connected by an
in-process queue: there is no socket, no port, no second machine, no discovery,
no Tor, and nothing survives the process unless ``--data-dir`` is given. Anything
said about anonymity, or about two real clients, rests on code that does not
exist yet -- see docs/SECURITY_BOUNDARIES.md.

What the safety numbers are for
-------------------------------
The printed pair is the only defence against a substituted bundle, which X3DH
cannot catch on its own: an attacker who replaces the identity key *and* signs
the replacement passes every check this program makes. Comparing those digits
over a channel the attacker does not control is the whole of the protection.
Printing them makes that comparison the next step rather than an afterthought;
the program cannot do it itself, because it has no way to know who it is talking
to.

Where the code is
-----------------
Everything interesting is in the package, not here: :mod:`src.protocol.session`
owns the handshake and the ratchet, :mod:`src.p2p.relay` carries the frames,
:mod:`src.storage` persists the keys. This file is the role-playing layer and it
is thin on purpose -- a CLI that grew its own protocol logic would be a second
implementation, and the only untested one.
"""

import argparse
import os
import sys
import textwrap
from typing import List, Optional, Tuple

# Running the file directly puts examples/ on sys.path, not the repository root,
# so the package would not be importable without an install. After
# `pip install -e .` this branch is not taken.
if __package__ in (None, ""):  # pragma: no cover - depends on invocation
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nacl.public import PrivateKey  # noqa: E402
from nacl.signing import SigningKey  # noqa: E402

from src.crypto.double_ratchet import (  # noqa: E402
    PROTOCOL_VERSION,
    IncompatibleSessionError,
)
from src.crypto.prekey_store import (  # noqa: E402
    PreKeyStore,
    stocked_prekey_store,
)
from src.crypto.pq import (  # noqa: E402
    PQPreKeyStore,
    stocked_pq_store,
)
from src.crypto.x3dh import X3DH, X3DHResponder  # noqa: E402
from src.p2p.relay import Relay  # noqa: E402
from src.protocol.padding import DEFAULT_PROFILE, FRAME_OVERHEAD_MAX  # noqa: E402
from src.protocol.session import Session, accept, initiate  # noqa: E402
from src.protocol.verifier import safety_number  # noqa: E402
from src.protocol.wire import decode_prekey_bundle, encode_prekey_bundle  # noqa: E402
from src.storage.container import Container, KdfParams, StorageError  # noqa: E402
from src.storage.store import (  # noqa: E402
    IdentityStore,
    PeerDirectory,
    PersistentPQPreKeyStore,
    PersistentPreKeyStore,
    SessionStore,
    peer_id,
)

#: One-time prekeys for a fresh identity. Signal uses a hundred or so. The pool
#: exists so the *next* handshake still has forward secrecy, and a demo that
#: runs a few times does not need more.
DEMO_PREKEYS = 10

#: The smallest frame a padded message can produce: the plaintext floor plus the
#: frame's own overhead. Used to spot a frame that escaped padding.
MIN_PADDED_FRAME = DEFAULT_PROFILE.floor + FRAME_OVERHEAD_MAX + 4

#: Container KDF cost. The library default is deliberately slow, which is right
#: for a real profile and wrong for a command that should start instantly. This
#: is a demo setting and changes nothing in the library.
DEMO_KDF = KdfParams(n_log2=14, r=8, p=1)

SENDER = "alice"
RECEIVER = "bob"


def rule(title: str = "") -> None:
    if not title:
        print("-" * 68)
        return
    print(f"\n-- {title} " + "-" * max(4, 66 - len(title)))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class EphemeralIdentity:
    """
    The three properties of :class:`~src.storage.store.IdentityStore`, in memory.

    So the demo path and the persistent path hand the same object to
    :class:`Participant`, and the role logic cannot tell them apart. Without
    this the two paths would differ in more than where the bytes live, and the
    difference would be invisible until one of them broke.
    """

    def __init__(
        self,
        identity_private: SigningKey,
        signed_prekey_private: PrivateKey,
        signed_prekey_id: int,
    ):
        self._identity = identity_private
        self._signed_prekey = signed_prekey_private
        self._signed_prekey_id = signed_prekey_id

    @property
    def identity_key(self) -> SigningKey:
        return self._identity

    @property
    def signed_prekey_private(self) -> PrivateKey:
        return self._signed_prekey

    @property
    def signed_prekey_id(self) -> int:
        return self._signed_prekey_id

    @property
    def epoch(self) -> int:
        return 1


def open_or_create(path: str, password: str) -> Container:
    """Open a container, creating it on first use."""
    if Container.exists(path):
        return Container.open(path, password)
    return Container.create(path, password, kdf_params=DEMO_KDF)


def identity_is_initialised(identity) -> bool:
    """
    Whether an :class:`IdentityStore` already holds keys.

    ``IdentityStore`` raises on an empty read rather than answering, and its
    ``initialise`` raises if called twice, so the state is worth asking about
    before either. A container that exists but was never filled in is a real
    case: ``Container.create`` succeeds, then the process can die before the
    first store.
    """
    try:
        identity.identity_key
    except StorageError:
        return False
    return True


def open_participant(
    name: str, directory: Optional[str], password: str
) -> "Participant":
    """
    Load a participant's keys, creating them on first run.

    With no directory, keys are generated per process and thrown away, which is
    the right default for a demo: nothing is written that was not asked for.
    With one, the identity, the signed prekey and both prekey pools are sealed
    in containers, so a second run is the same person talking to the same person.

    Both pools matter for the same reason and are handled identically. A pool
    that does not survive a restart is an empty pool, and an empty pool means
    every later handshake either fails or is accepted without post-quantum
    protection -- which is why the post-quantum pool is persisted rather than
    quietly regenerated.
    """
    if directory is None:
        x3dh = X3DH()
        identity_private, _ = x3dh.generate_identity_keys()
        return Participant(
            name,
            EphemeralIdentity(identity_private, PrivateKey.generate(), 1),
            stocked_prekey_store(DEMO_PREKEYS),
            stocked_pq_store(DEMO_PREKEYS),
        )

    os.makedirs(directory, exist_ok=True)
    identity = IdentityStore(
        open_or_create(os.path.join(directory, f"{name}-identity.bin"), password)
    )
    prekeys = PersistentPreKeyStore(
        open_or_create(os.path.join(directory, f"{name}-prekeys.bin"), password)
    )
    pq_prekeys = PersistentPQPreKeyStore(
        open_or_create(os.path.join(directory, f"{name}-pqkeys.bin"), password)
    )
    if not identity_is_initialised(identity):
        identity.initialise()
    if prekeys.count() == 0:
        prekeys.replenish(0, DEMO_PREKEYS)
    if pq_prekeys.count() == 0:
        pq_prekeys.replenish(0, DEMO_PREKEYS)
    return Participant(name, identity, prekeys, pq_prekeys)


# ---------------------------------------------------------------------------
# The two roles
# ---------------------------------------------------------------------------


class Participant:
    """
    One side of the demo: its keys, its responder, its session.

    Holds no protocol logic. Both roles are the same class, because the
    difference between them is which session function gets called first, and
    splitting them into two classes would hide that.
    """

    def __init__(
        self, name: str, identity, prekeys: PreKeyStore, pq_prekeys: PQPreKeyStore
    ):
        self.name = name
        self._identity = identity
        self.session: Optional[Session] = None
        self._responder = X3DHResponder(
            identity_private=identity.identity_key,
            signed_prekey_private=identity.signed_prekey_private,
            prekey_store=prekeys,
            pq_prekey_store=pq_prekeys,
            signed_prekey_id=identity.signed_prekey_id,
        )
        self._pq_prekeys = pq_prekeys

    @property
    def pq_prekey_pool_size(self) -> int:
        return self._pq_prekeys.count()

    @property
    def identity_public(self) -> SigningKey:
        return self._identity.identity_key.verify_key

    @property
    def prekey_pool_size(self) -> int:
        return self._responder.prekey_store.count()

    def as_receiver_publish_bundle(self) -> bytes:
        """
        The receiving role: publish a bundle for the sender to fetch.

        The one-time prekey is reserved while it is advertised, so a second
        bundle cannot hand out the same one.
        """
        bundle = self._responder.publish_bundle()
        frame = encode_prekey_bundle(bundle)
        rule("bundle")
        print(
            f"{self.name} published signed prekey "
            f"id {bundle.signed_prekey_id}, {len(frame)} byte frame"
        )
        print(
            f"{self.name} reserved one-time prekey "
            f"{bundle.one_time_prekey_id}; the pool holds "
            f"{self.prekey_pool_size} more"
        )
        print(
            f"{self.name} reserved post-quantum prekey "
            f"{bundle.pq_prekey_id} (ML-KEM-768, 1184 bytes of the frame); "
            f"the pool holds {self.pq_prekey_pool_size} more"
        )
        return frame

    def as_receiver_answer(self, init_frame: bytes) -> bytes:
        """The receiving role again: authenticate, spend the prekeys, confirm."""
        self.session, confirmation = accept(self._responder, init_frame)
        print(
            f"{self.name} authenticated the initiator and spent prekey "
            f"{self.session.one_time_prekey_id} and post-quantum prekey "
            f"{self.session.pq_prekey_id}"
        )
        print(
            f"{self.name} pools now hold {self.prekey_pool_size} and "
            f"{self.pq_prekey_pool_size}"
        )
        print(f"{self.name} sent a key confirmation derived from the root key")
        return confirmation

    def as_sender_start(self, bundle_frame: bytes) -> bytes:
        """
        The sending role: authenticate the bundle, produce the initial frame.

        The signature is checked before any of the bundle's keys is used, so a
        substituted signed prekey raises instead of quietly producing a session
        the attacker holds too.
        """
        self.session, init_frame = initiate(
            self._identity.identity_key, decode_prekey_bundle(bundle_frame)
        )
        rule("handshake")
        print(f"{self.name} verified the bundle signature before using it")
        print(
            f"{self.name} encapsulated against the post-quantum prekey, so "
            f"forward secrecy survives a future break of X25519"
        )
        print(
            f"{self.name} is not confirmed yet: the root key stays locked "
            f"until the reply verifies"
        )
        return init_frame

    def as_sender_confirm(self, confirmation: bytes) -> None:
        """The sending role again: verify the confirmation, unlock the root key."""
        self.session.confirm(confirmation)
        print(f"{self.name} verified the confirmation; the root key is now " f"usable")


# ---------------------------------------------------------------------------
# Talking
# ---------------------------------------------------------------------------


def converse(
    sender: Participant,
    receiver: Participant,
    relay: Relay,
    sender_endpoint,
    receiver_endpoint,
    count: int,
) -> None:
    """A conversation in both directions, over the transport."""
    rule("messages")
    for index in range(count):
        for text, source, sink, source_ep, sink_ep in (
            (
                f"message {index} from {sender.name}",
                sender,
                receiver,
                sender_endpoint,
                receiver_endpoint,
            ),
            (
                f"reply {index} from {receiver.name}",
                receiver,
                sender,
                receiver_endpoint,
                sender_endpoint,
            ),
        ):
            frame = source.session.send(text.encode("utf-8"))
            source_ep.send(frame)
            assert sink.session.receive(sink_ep.receive()).decode("utf-8") == text
            print(
                f"{source.name:>7} -> {sink.name:<7} {len(frame):>6} byte "
                f"frame  {text}"
            )

    print()
    print("An empty and a large message, to show the frame is not the message:")
    for size in (1, 4096):
        frame = sender.session.send(b"x" * size)
        sender_endpoint.send(frame)
        assert receiver.session.receive(receiver_endpoint.receive()) == b"x" * size
        print(f"  {size:>6} bytes of plaintext -> {len(frame):>6} byte frame")
    print(
        f"  the padding floor is {DEFAULT_PROFILE.floor} bytes of plaintext, "
        f"so a frame is at least {MIN_PADDED_FRAME} bytes"
    )


def report_lengths(relay: Relay, first_message: int) -> None:
    """
    Print what a passive observer gets, before and after padding.

    Both numbers come from :class:`~src.p2p.relay.Transport`, which records
    every frame it carries -- the same view a network observer has. Printing
    them here rather than asserting them in prose is the point: the padded
    figure is a measurement, and if the padding stopped working this line would
    show it.

    Args:
        first_message: index in the observation of the first message frame, so
            the bundle and the two handshake frames are not counted as
            messages -- their sizes are fixed and public anyway.
    """
    rule("what an observer sees")
    frames = relay.observed_lengths()[first_message:]
    distinct = sorted(set(frames))
    print(
        f"{len(frames)} message frames carried, "
        f"{len(distinct)} distinct length(s): {distinct}"
    )

    unpadded = [size for size in frames if size < MIN_PADDED_FRAME]
    if unpadded:
        print(
            textwrap.fill(
                f"UNEXPECTED: {len(unpadded)} frame(s) below the padding floor of "
                f"{MIN_PADDED_FRAME} bytes. Padding is not being applied to "
                f"everything it should be.",
                width=68,
            )
        )
    else:
        print(
            textwrap.fill(
                "Every message frame is the same size, whatever the message was: "
                "the padded plaintext length is rounded up to a bucket, and the "
                "frame is sized from that. The one thing still visible is which "
                "bucket a message fell into.",
                width=68,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
    print(
        textwrap.fill(
            "Still visible, and not fixable here: how many messages there are, when "
            "they were sent, which direction they went, and who they are with. "
            "Padding moves the length signal only. Cover traffic and timing "
            "obfuscation need a network to exist first.",
            width=68,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )


# ---------------------------------------------------------------------------
# Persistence of the conversation
# ---------------------------------------------------------------------------


def resume_existing(
    directory: str, password: str, sender: Participant, receiver: Participant
) -> Optional[Tuple[Session, Session]]:
    """
    Reopen both halves of a stored conversation.

    Both roles are needed because the demo keeps one directory for the whole
    conversation. A real client holds one identity and many peers, and would
    reopen only its own half.

    Returns None on a first run, and also when the stored session cannot be used:
    an unreadable or out-of-date state means a new handshake is the right
    response, not a crash.
    """
    peers_path = os.path.join(directory, "peers.bin")
    if not Container.exists(peers_path):
        return None
    peers = PeerDirectory(Container.open(peers_path, password))
    sessions = SessionStore(os.path.join(directory, "sessions"), password)

    restored = []
    for local, peer in ((sender, receiver), (receiver, sender)):
        role = peers.role_of(peer.identity_public)
        if role is None:
            return None
        try:
            state = sessions.load(peer_id(peer.identity_public), PROTOCOL_VERSION)
        except (KeyError, IncompatibleSessionError):
            return None
        restored.append(
            Session.from_state(
                state,
                is_initiator=role,
                local_identity=local.identity_public,
                peer_identity=peer.identity_public,
            )
        )
    return restored[0], restored[1]


def save_session(directory: str, password: str, peer, session: Session) -> None:
    """Write one half of a conversation down: its role, its peer, its state."""
    os.makedirs(directory, exist_ok=True)
    peers = PeerDirectory(
        open_or_create(os.path.join(directory, "peers.bin"), password)
    )
    peers.set(peer, is_initiator=session.is_initiator)
    SessionStore(os.path.join(directory, "sessions"), password).save(
        peer_id(peer), session.state
    )


# ---------------------------------------------------------------------------
# The demo
# ---------------------------------------------------------------------------


def run_demo(messages: int, directory: Optional[str], password: str) -> int:
    relay = Relay()
    sender_endpoint, receiver_endpoint = relay.link(SENDER, RECEIVER)

    sender = open_participant(SENDER, directory, password)
    receiver = open_participant(RECEIVER, directory, password)

    rule()
    print(f"{SENDER:<10}{bytes(sender.identity_public).hex()}")
    print(f"{RECEIVER:<10}{bytes(receiver.identity_public).hex()}")
    print(f'{"":<10}session protocol version {PROTOCOL_VERSION}')

    resumed = (
        resume_existing(directory, password, sender, receiver)
        if directory is not None
        else None
    )
    if resumed is not None:
        sender.session, receiver.session = resumed
        print(
            "\na stored session for this pair was found and reopened; no "
            "handshake was needed"
        )
    else:
        bundle_frame = receiver.as_receiver_publish_bundle()
        sender_endpoint.send(bundle_frame)
        init_frame = sender.as_sender_start(receiver_endpoint.receive())
        sender_endpoint.send(init_frame)
        confirmation = receiver.as_receiver_answer(receiver_endpoint.receive())
        receiver_endpoint.send(confirmation)
        sender.as_sender_confirm(sender_endpoint.receive())

    rule("safety numbers")
    print("Compare these over a channel the attacker does not control. If they")
    print("differ, the bundle was substituted and this conversation is with")
    print("someone else, however correct every check above was.\n")
    print(str(safety_number(sender.identity_public, receiver.identity_public)))

    first_message = len(relay.observed)

    converse(sender, receiver, relay, sender_endpoint, receiver_endpoint, messages)
    report_lengths(relay, first_message)

    if directory is not None:
        save_session(directory, password, receiver.identity_public, sender.session)
        save_session(directory, password, sender.identity_public, receiver.session)
        print(f"\nstate written to {directory}")

    rule("what this is not")
    print(
        textwrap.fill(
            "Both roles run in one process over an in-process queue. There is no "
            "socket, no port, no second machine, no peer discovery and no Tor. "
            "Padding hides the length of a message and nothing else: the count, "
            "the timing, the direction and the peer are all still there, as the "
            "numbers above show. This is a demonstration that the protocol joins "
            "up, not a messenger.",
            width=68,
        )
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run both roles of one conversation in one process, over an "
            "in-process transport."
        ),
    )
    parser.add_argument(
        "-n",
        "--messages",
        type=int,
        default=3,
        help="message/reply pairs to exchange (default: 3)",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        help=(
            "directory for the sealed key containers and the stored session. "
            "Without it, keys are generated per run and nothing is written."
        ),
    )
    parser.add_argument(
        "--password",
        default="demo",
        help=(
            "password for those containers (default: 'demo', which is not a "
            "password and is here only so the demo runs unattended)"
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.messages < 0:
        parser.error("--messages cannot be negative")
    return run_demo(arguments.messages, arguments.data_dir, arguments.password)


if __name__ == "__main__":
    sys.exit(main())
