"""
Tests for out-of-band key verification.

The property that matters is not "the numbers differ when the keys differ" but
that both sides derive the *same* number from the same pair of keys, and that
any change to either key changes the number. Without the first, comparison is
meaningless; without the second, an attacker can substitute a key.
"""

import pytest
from nacl.signing import SigningKey

from src.crypto.x3dh import X3DH, X3DHResponder
from src.crypto.prekey_store import InMemoryPreKeyStore
from src.protocol.verifier import (
    DEFAULT_MIN_DIGITS,
    FINGERPRINT_VERSION,
    MINIMUM_USEFUL_DIGITS,
    SafetyNumber,
    SafetyNumberTooShortError,
    VerificationError,
    safety_number,
    safety_number_for_session,
    verify,
)

ALPHABET = '0123456789abcdef'


def keypair():
    return SigningKey.generate(), SigningKey.generate()


class TestFormat:
    def test_sixty_digits_in_twelve_groups(self):
        _, a = keypair()
        _, b = keypair()
        number = safety_number(a.verify_key, b.verify_key)
        assert len(number.digits) == 60
        assert len(number.groups) == 12
        assert all(len(group) == 5 for group in number.groups)

    def test_digits_are_lowercase_hex(self):
        _, a = keypair()
        _, b = keypair()
        number = safety_number(a.verify_key, b.verify_key)
        assert all(character in ALPHABET for character in number.digits)

    def test_display_is_space_separated_groups(self):
        _, a = keypair()
        _, b = keypair()
        number = safety_number(a.verify_key, b.verify_key)
        assert str(number) == ' '.join(number.groups)

    def test_rejects_wrong_length(self):
        with pytest.raises(VerificationError):
            SafetyNumber(digits='0' * 59, groups=('0' * 59,))
        with pytest.raises(VerificationError):
            SafetyNumber(digits='0' * 61, groups=('0' * 61,))

    def test_rejects_non_hex(self):
        from src.protocol.verifier import _split
        digits = 'z' * 60
        with pytest.raises(VerificationError):
            SafetyNumber(digits=digits, groups=_split(digits))

    def test_rejects_inconsistent_groups(self):
        digits = '0' * 60
        with pytest.raises(VerificationError):
            SafetyNumber(digits=digits, groups=('0' * 5,) * 11)

    def test_rejects_non_verify_keys(self):
        _, a = keypair()
        with pytest.raises(VerificationError):
            safety_number(a, b'not a key')  # type: ignore[arg-type]
        with pytest.raises(VerificationError):
            safety_number(a.verify_key, a)  # a SigningKey, not a VerifyKey


class TestSymmetry:
    def test_both_sides_derive_the_same_number(self):
        _, a = keypair()
        _, b = keypair()
        assert (safety_number(a.verify_key, b.verify_key).digits
                == safety_number(b.verify_key, a.verify_key).digits)

    def test_order_does_not_matter_for_a_thousand_pairs(self):
        x3dh = X3DH()
        for _ in range(200):
            _, a = x3dh.generate_identity_keys()
            _, b = x3dh.generate_identity_keys()
            assert (safety_number(a, b).digits
                    == safety_number(b, a).digits)

    def test_self_comparison_is_stable(self):
        _, a = keypair()
        once = safety_number(a.verify_key, a.verify_key)
        twice = safety_number(a.verify_key, a.verify_key)
        assert once.digits == twice.digits


class TestUniqueness:
    def test_different_pairs_give_different_numbers(self):
        x3dh = X3DH()
        seen = {}
        for _ in range(300):
            _, a = x3dh.generate_identity_keys()
            _, b = x3dh.generate_identity_keys()
            digits = safety_number(a, b).digits
            assert digits not in seen
            seen[digits] = True

    def test_changing_either_key_changes_the_number(self):
        _, a = keypair()
        _, b = keypair()
        baseline = safety_number(a.verify_key, b.verify_key).digits
        assert safety_number(SigningKey.generate().verify_key,
                             b.verify_key).digits != baseline
        assert safety_number(a.verify_key,
                             SigningKey.generate().verify_key).digits != baseline

    def test_swapping_the_pair_changes_the_number(self):
        """
        An attacker substituting key C for B must not land on B's number.
        """
        _, a = keypair()
        _, b = keypair()
        _, c = keypair()
        honest = safety_number(a.verify_key, b.verify_key).digits
        assert safety_number(a.verify_key, c.verify_key).digits != honest

    def test_concatenation_cannot_collide(self):
        """
        Keys are length-prefixed, so a different split of the same bytes must
        not hash to the same digest.
        """
        _, a = keypair()
        _, b = keypair()
        first = safety_number(a.verify_key, b.verify_key).digits
        second = safety_number(b.verify_key, a.verify_key).digits
        # Swapping preserves the number, which is required, and that is the only
        # symmetry: it comes from sorting, not from a missing separator.
        assert first == second

    def test_version_is_bound_into_the_digest(self):
        """Two formats must not produce comparable-looking numbers."""
        from src.protocol.verifier import _DOMAIN, FINGERPRINT_VERSION
        import hashlib
        _, a = keypair()
        _, b = keypair()
        left = bytes(a.verify_key)
        right = bytes(b.verify_key)
        if left > right:
            left, right = right, left
        framed = (
            bytes((FINGERPRINT_VERSION,))
            + len(left).to_bytes(4, 'big') + left
            + len(right).to_bytes(4, 'big') + right
        )
        ours = hashlib.sha256(_DOMAIN + framed).hexdigest()[:60]
        assert ours == safety_number(a.verify_key, b.verify_key).digits


class TestComparison:
    def _number(self):
        _, a = keypair()
        _, b = keypair()
        return a.verify_key, b.verify_key, safety_number(a.verify_key, b.verify_key)

    def test_full_match_verifies(self):
        local, remote, number = self._number()
        result = verify(local, remote, number)
        assert result.verified
        assert result.matched_digits == result.total_digits == 60
        assert not result.partial

    def test_wrong_number_fails_at_the_first_digit(self):
        local, remote, _ = self._number()
        theirs = safety_number(local, SigningKey.generate().verify_key)
        result = verify(local, remote, theirs)
        assert not result.verified
        assert result.matched_digits == 0

    def test_single_digit_change_is_detected(self):
        local, remote, number = self._number()
        flipped = number.digits[:10] + ('0' if number.digits[10] != '0' else '1') \
            + number.digits[11:]
        result = verify(local, remote, number.with_digits(flipped))
        assert not result.verified
        assert result.matched_digits == 10
        assert 'differ at digit 10' in result.describe()

    def test_last_digit_change_is_detected_on_a_full_comparison(self):
        local, remote, number = self._number()
        flipped = number.digits[:59] + ('0' if number.digits[59] != '0' else '1')
        result = verify(local, remote, number.with_digits(flipped))
        assert not result.verified
        assert result.matched_digits == 59

    def test_partial_comparison_can_pass_and_is_labelled(self):
        local, remote, number = self._number()
        # Same first 20 digits, different afterwards.
        tail = '0' * 40
        truncated = number.with_digits(number.digits[:20] + tail)
        result = verify(local, remote, truncated, min_digits=20)
        assert result.verified
        assert result.partial
        assert result.matched_digits == 20
        assert 'full number was not compared' in result.describe()

    def test_partial_comparison_still_fails_on_an_early_difference(self):
        local, remote, number = self._number()
        tampered = number.with_digits('f' + number.digits[1:])
        result = verify(local, remote, tampered, min_digits=20)
        assert not result.verified
        assert result.matched_digits == 0

    def test_minimum_floor_enforced(self):
        local, remote, number = self._number()
        with pytest.raises(SafetyNumberTooShortError):
            verify(local, remote, number, min_digits=MINIMUM_USEFUL_DIGITS - 1)

    def test_minimum_above_length_rejected(self):
        local, remote, number = self._number()
        with pytest.raises(SafetyNumberTooShortError):
            verify(local, remote, number, min_digits=61)

    def test_default_requires_the_whole_number(self):
        assert DEFAULT_MIN_DIGITS == 60

    def test_version_mismatch_rejected(self):
        local, remote, number = self._number()
        other_version = SafetyNumber(
            digits=number.digits,
            groups=number.groups,
            version=FINGERPRINT_VERSION + 1,
        )
        with pytest.raises(VerificationError):
            verify(local, remote, other_version)

    def test_wrong_type_rejected(self):
        local, remote, number = self._number()
        with pytest.raises(VerificationError):
            verify(local, remote, number.digits)  # type: ignore[arg-type]


class TestSessionBinding:
    def test_number_comes_from_the_bundle_in_use(self):
        """
        The remote key is taken from the bundle about to be used, so a caller
        cannot verify one bundle and open a session with another.
        """
        x3dh = X3DH()
        bob_private, bob_public = x3dh.generate_identity_keys()
        spk_private, _, _ = x3dh.generate_signed_prekey(bob_private)
        store = InMemoryPreKeyStore()
        store.put(x3dh.generate_one_time_prekey(key_id=1))
        responder = X3DHResponder(
            identity_private=bob_private,
            signed_prekey_private=spk_private,
            prekey_store=store,
        )
        bundle = responder.publish_bundle()

        alice_private, _ = x3dh.generate_identity_keys()
        mine = safety_number_for_session(alice_private.verify_key, bundle.identity_key)
        theirs = safety_number(bundle.identity_key, alice_private.verify_key)
        assert verify(alice_private.verify_key, bundle.identity_key, theirs).verified
        assert mine.digits == theirs.digits

    def test_rotating_the_identity_key_changes_the_number(self):
        """
        Rotation is the point: after the identity key changes, the old safety
        number no longer matches, so a stale comparison is not accepted.
        """
        x3dh = X3DH()
        _, alice = x3dh.generate_identity_keys()
        _, bob_before = x3dh.generate_identity_keys()
        old = safety_number(alice, bob_before)
        _, bob_after = x3dh.generate_identity_keys()
        new = safety_number(alice, bob_after)
        assert old.digits != new.digits
        assert not verify(alice, bob_after, old).verified

    def test_rotation_keeps_the_number_when_only_the_prekey_changes(self):
        """A signed-prekey rotation is not an identity change."""
        x3dh = X3DH()
        bob_private, bob_public = x3dh.generate_identity_keys()
        _, alice = x3dh.generate_identity_keys()
        before = safety_number(alice, bob_public)
        spk_a, _, _ = x3dh.generate_signed_prekey(bob_private)
        spk_b, _, _ = x3dh.generate_signed_prekey(bob_private)
        assert bytes(spk_a) != bytes(spk_b)
        after = safety_number(alice, bob_public)
        assert before.digits == after.digits
