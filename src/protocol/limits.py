"""
Limits that more than one protocol module has to agree on.

They lived in :mod:`src.protocol.wire`, which turned out to be the wrong place
the moment a second module needed one of them: :mod:`src.protocol.padding`
computes the largest plaintext that still fits in a frame, so it needs
``MAX_FRAME_BYTES``, and importing it from ``wire`` while ``wire`` imports the
padding profile from ``padding`` is a cycle.

The alternative to this file was to duplicate the constant, or to pass it in as
a parameter everywhere. A duplicated limit is a limit that can disagree with
itself, and a limit that can disagree with itself is a limit that will.
"""

#: Refuse frames larger than this before allocating. Ratchet messages are small;
#: anything near this limit is an attack or a bug.
MAX_FRAME_BYTES = 1 << 20  # 1 MiB

#: Guard against absurd varints even inside a legal frame.
MAX_VARINT_SHIFT = 63

#: AEAD framing inside a ratchet message: a nonce in the clear and a
#: Poly1305 tag at the end. Both sit outside the message length, so any layer
#: that reasons about "how long is this message on the wire" has to count them.
#: They live here because the wire encoder and the padding profile both need
#: them, and two copies of a length that has to be exact is one copy too many.
NONCE_LEN = 24
TAG_LEN = 16
AEAD_OVERHEAD = NONCE_LEN + TAG_LEN
