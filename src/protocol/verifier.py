"""
Out-of-band key verification (safety numbers).

X3DH authenticates the *relationship* between an identity key and a signed
prekey, and the Double Ratchet authenticates the session once it exists. Neither
defends against a wholesale bundle substitution: an active attacker who replaces
the identity key **and** the signed prekey, signing the replacement with their
own identity key, produces a bundle that verifies perfectly. The initiator ends
up in a genuine, correctly encrypted session with the wrong person, and nothing
inside the protocol can tell.

The only defence is comparison out of band. Both sides derive a short number
from the same pair of identity keys and compare it over a channel the attacker
does not control: read aloud, scanned as a QR code, or shown in person. A
mismatch means a substitution, and the session must be abandoned.

Format, stated precisely so nobody has to reverse it:

- 30 bytes of SHA-256 digest give 60 nibbles, rendered as 60 hexadecimal
  characters in twelve groups of five.
- the two identity keys are sorted bytewise before hashing, so the result does
  not depend on which side computes it and no ordering has to be agreed on.
- each key is length-prefixed into the digest, so a different split of the same
  bytes cannot collide.
- a format version byte is mixed in, so numbers from two different derivations
  can never be compared against each other successfully.

This is **not** Signal's fingerprint format and is not compatible with it. Our
identity keys are Ed25519 converted to X25519, where Signal uses native
curve25519 identity keys, so the digests could not match even in principle.
Claiming compatibility would be false.
"""

import hashlib
from dataclasses import dataclass
from typing import Tuple

from nacl.signing import VerifyKey

#: Bumped if the derivation ever changes.
FINGERPRINT_VERSION = 1

_DOMAIN = b"anarchy-safety-number-v1"

#: Bytes of digest consumed: 30 bytes give 60 nibbles, one character each.
_DIGEST_BYTES = 30
_DIGITS = _DIGEST_BYTES * 2

_GROUP_SIZE = 5
_GROUPS = _DIGITS // _GROUP_SIZE

_HEX = '0123456789abcdef'

#: How many leading digits must match for a verification to pass. Kept at the
#: full number by default: a prefix match is a strong signal, not proof.
DEFAULT_MIN_DIGITS = _DIGITS

#: Below this many matching digits a comparison means nothing, whatever the
#: caller asked for.
MINIMUM_USEFUL_DIGITS = 8


class VerificationError(Exception):
    """Base class for verification failures."""


class SafetyNumberTooShortError(VerificationError):
    """The caller asked for fewer matching digits than can mean anything."""


@dataclass(frozen=True)
class SafetyNumber:
    """
    A derived safety number: 60 hexadecimal characters in twelve groups.

    ``digits`` is the whole string and ``groups`` the same value split for
    display. Equality of two numbers is what verification rests on.
    """

    digits: str
    groups: Tuple[str, ...]
    version: int = FINGERPRINT_VERSION

    def __post_init__(self) -> None:
        if len(self.digits) != _DIGITS:
            raise VerificationError(
                f'safety number must have {_DIGITS} digits, got {len(self.digits)}'
            )
        for character in self.digits:
            if character not in _HEX:
                raise VerificationError(
                    'safety number must be lowercase hexadecimal'
                )
        if tuple(self.groups) != _split(self.digits):
            raise VerificationError('groups do not match digits')

    def matches(self, other: 'SafetyNumber', digits: int = DEFAULT_MIN_DIGITS) -> int:
        """
        Count the leading digits that agree with ``other``.

        Returns:
            The number of matching leading digits, 0 if the very first differs.

        Raises:
            VerificationError: ``other`` uses a different format version.
        """
        if not isinstance(other, SafetyNumber):
            raise VerificationError('other must be a SafetyNumber')
        if other.version != self.version:
            raise VerificationError(
                f'version mismatch: {other.version} != {self.version}'
            )
        for index in range(min(digits, _DIGITS)):
            if self.digits[index] != other.digits[index]:
                return index
        return min(digits, _DIGITS)

    def with_digits(self, digits: str) -> 'SafetyNumber':
        """
        Return a copy carrying ``digits``.

        Useful for tests and for entering a scanned number by hand; the value
        still goes through the same validation.
        """
        return SafetyNumber(digits=digits, groups=_split(digits))

    def __str__(self) -> str:
        return ' '.join(self.groups)


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of comparing a locally derived number with one obtained out of band."""

    verified: bool
    matched_digits: int
    total_digits: int
    required_digits: int

    @property
    def partial(self) -> bool:
        """Matched, but on a prefix shorter than a full comparison."""
        return self.verified and self.required_digits < self.total_digits

    def describe(self) -> str:
        if not self.verified:
            return (
                f'keys do not match: they differ at digit {self.matched_digits} '
                f'of {self.total_digits}'
            )
        if self.partial:
            return (
                f'first {self.matched_digits} digits match, '
                f'the full number was not compared'
            )
        return f'keys match on all {self.total_digits} digits'


def _split(digits: str) -> Tuple[str, ...]:
    return tuple(
        digits[index:index + _GROUP_SIZE]
        for index in range(0, len(digits), _GROUP_SIZE)
    )


def _digest(identity_a: VerifyKey, identity_b: VerifyKey) -> bytes:
    """
    Hash two identity keys symmetrically.

    Sorting first is what makes the result independent of which side computes
    it. The keys are length-prefixed so that a different split of the same bytes
    cannot produce the same digest.
    """
    left = bytes(identity_a)
    right = bytes(identity_b)
    if left > right:
        left, right = right, left
    framed = (
        bytes((FINGERPRINT_VERSION,))
        + len(left).to_bytes(4, 'big') + left
        + len(right).to_bytes(4, 'big') + right
    )
    return hashlib.sha256(_DOMAIN + framed).digest()[:_DIGEST_BYTES]


def safety_number(identity_a: VerifyKey, identity_b: VerifyKey) -> SafetyNumber:
    """
    Derive the safety number for a pair of identity keys.

    Args:
        identity_a: one party's Ed25519 identity key.
        identity_b: the other party's. Order does not matter.

    Returns:
        A :class:`SafetyNumber` of 60 hexadecimal digits in twelve groups.
    """
    if not isinstance(identity_a, VerifyKey) or not isinstance(identity_b, VerifyKey):
        raise VerificationError('both arguments must be VerifyKey')

    digest = _digest(identity_a, identity_b)
    digits = ''.join(_HEX[byte >> 4] + _HEX[byte & 0x0F] for byte in digest)
    if len(digits) != _DIGITS:                 # pragma: no cover - arithmetic
        raise VerificationError('internal: wrong digit count')
    return SafetyNumber(digits=digits, groups=_split(digits))


def safety_number_for_session(
    local_identity: VerifyKey,
    bundle_identity: VerifyKey,
) -> SafetyNumber:
    """
    The number to compare before trusting ``bundle_identity``.

    Taking the remote key from the bundle that is about to be used keeps the two
    in step, so a caller cannot verify one bundle and then open a session with
    another.
    """
    return safety_number(local_identity, bundle_identity)


def verify(
    local_identity: VerifyKey,
    remote_identity: VerifyKey,
    theirs: SafetyNumber,
    min_digits: int = DEFAULT_MIN_DIGITS,
) -> VerificationResult:
    """
    Compare the locally derived number with one obtained out of band.

    Args:
        theirs: the number the other party read out, scanned or typed.
        min_digits: how many leading digits must agree. Defaults to a full
            comparison. A shorter value yields a *partial* result rather than a
            verification, because a prefix match is a much weaker statement.

    Raises:
        SafetyNumberTooShortError: ``min_digits`` is below the useful floor or
            above the number length.
    """
    if min_digits < MINIMUM_USEFUL_DIGITS:
        raise SafetyNumberTooShortError(
            f'min_digits must be at least {MINIMUM_USEFUL_DIGITS}, got {min_digits}'
        )
    if min_digits > _DIGITS:
        raise SafetyNumberTooShortError(f'min_digits cannot exceed {_DIGITS}')
    ours = safety_number(local_identity, remote_identity)
    matched = ours.matches(theirs, min_digits)
    return VerificationResult(
        verified=matched >= min_digits,
        matched_digits=matched,
        total_digits=_DIGITS,
        required_digits=min_digits,
    )
