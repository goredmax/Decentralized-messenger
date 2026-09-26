"""
HKDF (RFC 5869) over HMAC-SHA256.

Kept in one place on purpose. The X3DH root key and the Double Ratchet chain
keys are both derived with HKDF, and the two call sites previously carried
separate copies of it. One of those copies was wrong: it expanded to a single
32-byte HMAC and then read a second 32-byte half out of a 32-byte buffer, so
the chain key came back empty and every session derived the same message keys
from ``HMAC(b"", ...)``. Self-consistent on both sides, so no round-trip test
noticed.

Use :func:`hkdf` unless you specifically need the two stages.
"""

import hashlib
import hmac

HASH = hashlib.sha256
HASH_LEN = 32
MAX_OUTPUT = 255 * HASH_LEN


def extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract. An empty salt is treated as ``HASH_LEN`` zero bytes."""
    if not salt:
        salt = b'\x00' * HASH_LEN
    return hmac.new(salt, ikm, HASH).digest()


def expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    HKDF-Expand to ``length`` bytes.

    Raises:
        ValueError: ``length`` is negative or beyond RFC 5869's 255-block limit.
    """
    if length < 0 or length > MAX_OUTPUT:
        raise ValueError(
            f'HKDF output length must be 0..{MAX_OUTPUT}, got {length}'
        )
    okm = b''
    block = b''
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), HASH).digest()
        okm += block
        counter += 1
    return okm[:length]


def hkdf(salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Extract followed by HKDF-Expand."""
    return expand(extract(salt, ikm), info, length)
