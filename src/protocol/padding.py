"""
Length hiding: what an observer can count and what can be done about it.

The problem
-----------
A frame's length is visible to anyone on the path. A 40-byte message and a
4000-byte one are distinguishable without decrypting anything, which turns a
known-plaintext guess into a confirmation and, over enough messages, turns
traffic into a transcript of sizes. The Double Ratchet hides *content*; nothing
in it hides *size*.

Two layers, and why both
------------------------
1. **The plaintext is padded** (``pad``/``unpad``), inside the AEAD. The
   padding lives under the encryption, so the length of the real message is
   hidden from anyone who can decrypt but should not learn it -- a recipient
   forwarding a conversation onward, for instance.
2. **The frame is quantised** in :mod:`src.protocol.wire`, to a bucket taken
   from the same profile. This is the layer that does the work against a
   network observer, and it is deliberately outside the AEAD: the length of a
   ciphertext is public by construction.

Layer 2 also makes the frame length *exactly* constant within a bucket, rather
than constant to within the two bytes the header varints grow by. Those two
bytes leak nothing new -- ``pn`` and ``n`` travel in the clear and are readable
anyway -- but a length that wobbles by a byte every 128 messages is a strange
thing to hand to someone who is trying to measure you.

What the scheme does not hide
-----------------------------
- **The bucket.** Every padded message in a bucket is the same size, so the
  bucket is public. That is what a ladder is: it trades exact length for a
  coarse class. A single fixed size for all traffic would hide that too, at the
  cost of the bandwidth, and the profile makes that a choice.
- **The number of messages, and when they were sent.** Padding does nothing
  about either. Cover traffic and timing obfuscation would, and both need a
  network to exist first (docs/SECURITY_BOUNDARIES.md section 5).
- **Direction and peer.** Nothing here addresses either.
- **Traffic analysis by volume.** Padding every message to the same size
  removes one signal and can sharpen another: a stream of uniform frames is
  itself a signature.

The default profile
-------------------
Floor 8192, then steps of 8192. The floor is a deliberate choice rather than the
Signal ladder of 256-byte steps: a 256-byte step means a 1-byte message and a
5000-byte message land in different buckets, and the difference between their
frame lengths is exactly the leak this module exists to close. Below the floor
the size is not merely coarsened, it is gone. The price is that every short
message costs 8 KiB of ciphertext, and the profile is a parameter so a deployment
that would rather leak a coarse bucket can pay less.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from .limits import AEAD_OVERHEAD, MAX_FRAME_BYTES

#: The most a ratchet frame can add on top of the padded plaintext: version and
#: kind, the payload length varint, the cleartext header (a 32-byte DH key and
#: three varints), and the AEAD nonce and tag. Every varint is counted at its
#: widest, so a frame padded against this constant is never short of room, and
#: the frame length does not wobble when a counter's varint grows.
#:
#: It is deliberately an over-estimate. The alternative is to compute the
#: padding from the actual varint widths, which makes the frame length vary by a
#: byte every time a counter crosses 128. Those bytes are not secret -- ``pn`` and
#: ``n`` travel in the clear -- but a length that wobbles on a schedule is a
#: strange thing to hand to anyone measuring you.
FRAME_OVERHEAD_MAX = 2 + 9 + 32 + 9 + 9 + 9 + AEAD_OVERHEAD

#: Room reserved in a frame for everything that is not padding.
RESERVED_BYTES = FRAME_OVERHEAD_MAX

#: The largest padded plaintext that still fits in a frame.
MAX_BUCKET = MAX_FRAME_BYTES - RESERVED_BYTES


class PaddingError(Exception):
    """
    A padding layer could not be applied or undone.

    Distinct from a decryption failure on purpose. A ciphertext that decrypted
    and then failed to unpad means the padding itself was wrong, which is a bug
    or a downgrade attempt -- not a corrupted packet, which would have failed at
    the AEAD first.
    """


#: Room reserved for everything that is not padding: version and kind, the
#: length varint, the cleartext header, the ciphertext length varint, and the
#: AEAD nonce and tag. Worst case for every varint at once, so a bucket computed
#: against this can never overflow MAX_FRAME_BYTES.
RESERVED_BYTES = FRAME_OVERHEAD_MAX

#: The largest bucket that still fits in a frame.
MAX_BUCKET = MAX_FRAME_BYTES - RESERVED_BYTES


@dataclass(frozen=True)
class PaddingProfile:
    """
    A size ladder, for the plaintext and for the frame that carries it.

    Args:
        floor: every message is padded to at least this many plaintext bytes.
            Below the floor, length is not coarsened but hidden. A floor of zero
            disables padding, which is useful only for measuring what padding
            costs.
        step: the width of each bucket above the floor.

    The frame ladder is the plaintext ladder shifted up by
    :data:`FRAME_OVERHEAD_MAX`. The shift is what stops the two layers from
    doubling every message: a frame is always bigger than the plaintext inside
    it, so a frame measured on the *same* ladder as its plaintext lands in the
    next bucket and costs a second step. Shifting the ladder means a short
    message costs one bucket, not two.
    """

    floor: int = 8192
    step: int = 8192

    def __post_init__(self) -> None:
        for name in ("floor", "step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise PaddingError(f"{name} must be an int")
            if value < 0:
                raise PaddingError(f"{name} must not be negative")
        if self.step < 1:
            raise PaddingError(f"step must be positive, got {self.step}")
        if self.floor > MAX_BUCKET:
            raise PaddingError(
                f"floor {self.floor} does not fit in a frame (max {MAX_BUCKET})"
            )
        if self.step > MAX_FRAME_BYTES:
            raise PaddingError(
                f"step {self.step} does not fit in a frame " f"(max {MAX_FRAME_BYTES})"
            )

    @property
    def enabled(self) -> bool:
        """Whether this profile pads at all."""
        return self.floor > 0

    def _ladder(self, length: int, floor: int) -> int:
        if length <= floor:
            return floor
        excess = length - floor
        steps = -(-excess // self.step)  # ceiling division
        target = floor + steps * self.step
        if target > MAX_FRAME_BYTES - 2:
            raise PaddingError(
                f"padded to {target}, which does not fit in a frame "
                f"(max {MAX_FRAME_BYTES - 2})"
            )
        return target

    def bucket(self, length: int) -> int:
        """
        The padded size for a plaintext of ``length`` bytes.

        Monotonic by construction, so two messages whose padded sizes are equal
        really were indistinguishable to an observer.

        ``length`` is the whole padded plaintext, length prefix included, which
        is what :func:`pad` passes in. The consequence is that the floor is a
        bound on the padded size and not quite on the message: a message of
        ``floor - 1`` bytes needs a two-byte prefix and so lands in the bucket
        above. That is a one-value-wide seam at the floor, it is the price of
        making the padded size depend on the message length and nothing else,
        and tests/test_padding.py pins the exact boundary rather than leaving it
        to be discovered.
        """
        if isinstance(length, bool) or not isinstance(length, int):
            raise PaddingError("length must be an int")
        if length < 0:
            raise PaddingError(f"length must not be negative, got {length}")
        if not self.enabled:
            return length
        return self._ladder(length, self.floor)

    def frame_payload_length(self, content: int) -> int:
        """
        The payload length a frame should have to carry ``content`` bytes.

        ``content`` is everything the frame holds apart from the two-byte frame
        header: the cleartext header, the ciphertext and any fill.

        The result is on the shifted ladder, so a frame's total length depends
        only on which bucket its plaintext fell into -- and not on the counter
        values, whose varints grow as the conversation goes on.
        """
        if isinstance(content, bool) or not isinstance(content, int):
            raise PaddingError("content must be an int")
        if content < 0:
            raise PaddingError(f"content must not be negative, got {content}")
        if not self.enabled:
            return content
        return self._ladder(content, self.floor + FRAME_OVERHEAD_MAX)

    def max_plaintext(self) -> int:
        """Largest message this profile can carry, before padding."""
        return MAX_BUCKET

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        return f"floor {self.floor}, step {self.step}"


#: The profile in force unless a caller asks for another one.
DEFAULT_PROFILE = PaddingProfile()


def _varint(value: int) -> bytes:
    # A local copy rather than an import from wire: padding is applied before a
    # frame exists, and importing the frame encoder here would make the padding
    # layer depend on the layer it is trying to keep short. The encoding is
    # canonical LEB128, the same one wire.py writes, and test_padding.py pins
    # the two against each other.
    if value < 0:
        raise PaddingError("varint must not be negative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    start = offset
    last = 0
    while True:
        if offset >= len(buf):
            raise PaddingError("padding length runs past the end of the input")
        last = buf[offset]
        offset += 1
        result |= (last & 0x7F) << shift
        if not last & 0x80:
            break
        shift += 7
        if shift > 63:
            raise PaddingError("padding length is longer than 64 bits")
    if offset - start > 1 and last == 0x00:
        raise PaddingError("padding length is not a canonical varint")
    return result, offset


def pad(plaintext: bytes, profile: PaddingProfile = DEFAULT_PROFILE) -> bytes:
    """
    Pad a plaintext to its bucket: ``length || plaintext || random``.

    The length prefix is what makes the padding unambiguous. Without it a
    receiver could not tell how much of the decrypted blob was message, and any
    scheme that guessed would be guessable.

    The filler is random, not zeros. Zeros would make the padding
    distinguishable from data that happens to contain zeros, which is a smaller
    leak than the length but is still a free one to give up.

    A profile with a floor of zero returns the message unchanged, so the padding
    can be switched off without a second code path.

    Raises:
        PaddingError: the message does not fit in a frame once padded.
    """
    if not isinstance(plaintext, (bytes, bytearray, memoryview)):
        raise PaddingError("plaintext must be bytes")
    plaintext = bytes(plaintext)
    if not profile.enabled:
        return plaintext
    header = _varint(len(plaintext))
    target = profile.bucket(len(plaintext) + len(header))
    filler_length = target - len(header) - len(plaintext)
    if filler_length < 0:  # pragma: no cover - bucket() guarantees this
        raise PaddingError("bucket smaller than the message it holds")
    return header + plaintext + os.urandom(filler_length)


def unpad(padded: bytes) -> bytes:
    """
    Recover the message from a padded plaintext.

    Raises:
        PaddingError: the padding is malformed -- no length prefix, a length
            prefix that is not canonical, or a declared length that does not fit
            inside the input. A truncated or tampered ciphertext cannot reach
            here: the AEAD rejects it first.
    """
    if not isinstance(padded, (bytes, bytearray, memoryview)):
        raise PaddingError("padded plaintext must be bytes")
    padded = bytes(padded)
    if not padded:
        raise PaddingError("padded plaintext is empty")
    length, offset = _read_varint(padded, 0)
    if length > len(padded) - offset:
        raise PaddingError(
            f"padded plaintext declares {length} bytes of message but only "
            f"{len(padded) - offset} follow the prefix"
        )
    return padded[offset : offset + length]


def fits(
    plaintext_length: int, profile: PaddingProfile = DEFAULT_PROFILE
) -> Optional[str]:
    """
    Why a message of this length cannot be sent, or None if it can.

    A separate question from :func:`pad` so a caller can ask before building a
    message rather than after, and so the error can name the limit.
    """
    try:
        profile.bucket(plaintext_length + len(_varint(plaintext_length)))
    except PaddingError as exc:
        return str(exc)
    return None
