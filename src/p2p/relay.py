"""
Transport interface, and an in-process relay that implements it.

This is the seam a real network will be spliced into, so it is defined as an
interface first and a working implementation second. Anything that satisfies
:class:`Transport` can carry a session: a relay between two halves of one
process, a TCP connection, a libp2p stream, a Tor circuit. The alternative --
handing sessions a socket directly -- would put the choice of network inside the
protocol layer, and the protocol layer is the part that has to stay right.

Why the relay records frame lengths
-----------------------------------
Frame length is the metadata this project cannot currently hide, because
padding does not exist yet. An in-process relay sees exactly what a network
observer would see: the size of every frame, in order, with nothing added on
the way. That makes it the natural place to measure, and the measurement is
what :mod:`src.protocol.padding` will later be judged by. A padding scheme
that does not move the length a relay reports is not padding.

What the relay is not
---------------------
It is not a network. Two clients in one process can exchange bytes; two clients
in two processes cannot, because nothing here crosses a process boundary. Any
claim about a running messenger, about two machines, or about anonymity rests on
code that does not exist yet. See docs/SECURITY_BOUNDARIES.md section 5.
"""

import abc
import queue
import threading
from typing import Dict, List, Optional, Tuple

from ..protocol.wire import MAX_FRAME_BYTES

#: Frames a direction will hold before :meth:`Endpoint.send` refuses.
#:
#: Bounded because an unbounded buffer is a denial of service with no attacker:
#: a peer that sends faster than the other side reads grows the queue until the
#: process dies. A full queue makes the sender fail while the receiver keeps its
#: session state, which is recoverable; an unbounded queue makes the process
#: unrecoverable.
DEFAULT_QUEUE_SIZE = 256


class TransportError(Exception):
    """Base class for transport failures."""


class FrameTooLarge(TransportError):
    """A frame exceeds what the transport is willing to carry."""


class QueueFull(TransportError):
    """The peer's inbound queue is full; the frame was not accepted."""


class NoFrame(TransportError):
    """No frame arrived before the timeout expired."""


class Closed(TransportError):
    """The transport is closed."""


class Transport(abc.ABC):
    """
    A bidirectional, ordered, reliable byte channel.

    Reliability and ordering are part of the interface, not an implementation
    detail. The Double Ratchet tolerates reordering up to a bounded window and
    Sender Keys do not tolerate it at all, so a transport that silently drops or
    reorders frames would turn a protocol guarantee into a coin flip. Anything
    weaker than this has to say so in its own name.
    """

    @abc.abstractmethod
    def send(self, frame: bytes) -> None:
        """
        Hand one complete frame to the peer.

        Raises:
            FrameTooLarge: the frame exceeds the transport's size limit.
            QueueFull: the peer is not keeping up.
            Closed: the transport is closed.
        """

    @abc.abstractmethod
    def receive(self, timeout: Optional[float] = None) -> bytes:
        """
        Take the next frame from the peer, or wait for one.

        Args:
            timeout: seconds to wait, or None to wait indefinitely.

        Raises:
            NoFrame: the timeout expired.
            Closed: the transport is closed.
        """


class Endpoint(Transport):
    """One side of a relay link, from one participant's point of view."""

    def __init__(self, relay: "Relay", name: str, peer: str, max_frame_bytes: int):
        self._relay = relay
        self._name = name
        self._peer = peer
        self._max_frame_bytes = max_frame_bytes

    @property
    def name(self) -> str:
        """This end's name, as given to :meth:`Relay.link`."""
        return self._name

    @property
    def peer(self) -> str:
        """The name of the other end."""
        return self._peer

    def send(self, frame: bytes) -> None:
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise TransportError("a frame must be bytes")
        frame = bytes(frame)
        if len(frame) > self._max_frame_bytes:
            # Checked before the frame is queued, so an oversized frame costs
            # nothing beyond the caller's own memory for it.
            raise FrameTooLarge(
                f"frame of {len(frame)} bytes exceeds the {self._max_frame_bytes} "
                f"byte limit"
            )
        self._relay._deliver(self._name, self._peer, frame)

    def receive(self, timeout: Optional[float] = None) -> bytes:
        return self._relay._collect(self._name, timeout)

    def close(self) -> None:
        self._relay._close(self._name)

    def __enter__(self) -> "Endpoint":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Endpoint {self._name} -> {self._peer}>"


class Relay:
    """
    An in-process transport between named endpoints.

    Deliberately synchronous and single-process. Its job is to be the smallest
    thing that behaves like a network in the ways that matter for testing:
    frames arrive whole, in order, and their sizes are observable.

    Observability is the point. Every frame that goes through is appended to
    :attr:`observed`, as ``(sender, length)``. That is the same view a passive
    network observer has, which makes it the thing to assert on when asking
    whether a message length is hidden. It is not the total frame size on the
    wire -- a real network adds headers this relay does not -- but it is the
    part the protocol is responsible for.
    """

    def __init__(
        self,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ):
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if max_frame_bytes < 1:
            raise ValueError("max_frame_bytes must be at least 1")
        self._queue_size = queue_size
        self._max_frame_bytes = max_frame_bytes
        self._queues: Dict[str, "queue.Queue"] = {}
        self._endpoints: Dict[str, Endpoint] = {}
        self._links: Dict[str, str] = {}
        self._observed: List[Tuple[str, int]] = []
        self._closed: set = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def link(self, first: str, second: str) -> Tuple[Endpoint, Endpoint]:
        """
        Connect two names and return their endpoints.

        Args:
            first: the first participant's name.
            second: the second participant's name.

        Returns:
            ``(endpoint_for_first, endpoint_for_second)``.

        Raises:
            TransportError: a name is already linked, or a name is linked to
                itself. A participant that could reach itself would be able to
                read its own inbound queue, which no real network allows.
        """
        if first == second:
            raise TransportError("an endpoint cannot be linked to itself")
        with self._lock:
            for name in (first, second):
                if name in self._endpoints:
                    raise TransportError(f"{name!r} is already linked")
            endpoints = tuple(
                Endpoint(self, name, peer, self._max_frame_bytes)
                for name, peer in ((first, second), (second, first))
            )
            self._endpoints[first] = endpoints[0]
            self._endpoints[second] = endpoints[1]
            for name in (first, second):
                self._queues[name] = queue.Queue(self._queue_size)
            self._links[first] = second
            self._links[second] = first
        return endpoints[0], endpoints[1]

    def participants(self) -> List[str]:
        """Every linked name, in the order it was linked."""
        with self._lock:
            return list(self._endpoints)

    def is_closed(self, name: str) -> bool:
        with self._lock:
            return name in self._closed

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def observed(self) -> Tuple[Tuple[str, int], ...]:
        """
        Every frame that has passed through, as ``(sender, length)``.

        A copy, so a caller holding it cannot be surprised by later traffic, and
        so a test cannot accidentally "record" a frame by reading this.
        """
        with self._lock:
            return tuple(self._observed)

    def observed_lengths(self, sender: Optional[str] = None) -> List[int]:
        """Frame lengths, optionally for one sender only."""
        with self._lock:
            return [
                length
                for name, length in self._observed
                if sender is None or name == sender
            ]

    def reset_observation(self) -> None:
        """Forget the recorded frames, keeping the queues."""
        with self._lock:
            self._observed.clear()

    # ------------------------------------------------------------------
    # Internals, called by Endpoint
    # ------------------------------------------------------------------

    def _deliver(self, sender: str, recipient: str, frame: bytes) -> None:
        # One lock, and the record is appended after the enqueue succeeds. Both
        # halves of that matter: a frame the queue refused was never sent, so
        # recording it would make the observed log disagree with what the peer
        # actually received, and recording outside the lock would let two threads
        # append out of order, which would quietly make the log useless for the
        # one thing it is for.
        #
        # put_nowait does not block, so holding the lock across it costs
        # nothing and keeps the observation strictly ordered.
        with self._lock:
            if sender in self._closed or recipient in self._closed:
                raise Closed("the transport is closed")
            target = self._queues.get(recipient)
            if target is None:
                raise Closed(f"{recipient!r} has no link")
            try:
                target.put_nowait(frame)
            except queue.Full:
                raise QueueFull(
                    f"{recipient!r} is not reading; {self._queue_size} frames "
                    f"are already queued"
                ) from None
            self._observed.append((sender, len(frame)))

    def _collect(self, name: str, timeout: Optional[float]) -> bytes:
        with self._lock:
            if name in self._closed:
                raise Closed("the transport is closed")
            source = self._queues.get(name)
        if source is None:
            raise Closed(f"{name!r} has no link")
        try:
            return source.get(timeout=timeout)
        except queue.Empty:
            raise NoFrame(
                f'no frame from {self._links.get(name, "?")!r} within '
                f"{timeout} seconds"
            ) from None

    def _close(self, name: str) -> None:
        with self._lock:
            self._closed.add(name)
            self._endpoints.pop(name, None)
            self._queues.pop(name, None)
            self._links.pop(name, None)

    def close(self) -> None:
        """Close every endpoint."""
        with self._lock:
            for name in list(self._endpoints):
                self._closed.add(name)
            self._endpoints.clear()
            self._queues.clear()
            self._links.clear()
