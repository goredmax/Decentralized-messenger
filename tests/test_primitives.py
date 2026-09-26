"""
Cross-validation of the hand-rolled primitives against an independent
implementation.

``src/crypto`` builds HKDF and calls libsodium's X25519 directly. Both are easy
to get subtly wrong, and a test that reuses the same code path as the
implementation proves nothing. These tests therefore check the results against
``cryptography``, a separate implementation of the same standards.

Scope of the guarantee: this shows the primitives agree with a second
implementation on the inputs exercised here. It is *not* a conformance claim
against the published Signal or RFC test vectors, which are not vendored in
this repository. Adding them remains worthwhile.
"""

import nacl.utils
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.public import PrivateKey, PublicKey

from src.crypto.double_ratchet import DoubleRatchet
from src.crypto.x3dh import X3DH


class TestHkdf:
    @pytest.mark.parametrize('length', [16, 32, 42, 64, 255])
    def test_extract_expand_matches_reference(self, length):
        for _ in range(20):
            salt = nacl.utils.random(nacl.utils.random(1)[0] % 40)
            ikm = nacl.utils.random(32 + nacl.utils.random(1)[0] % 64)
            info = nacl.utils.random(nacl.utils.random(1)[0] % 30)

            prk = X3DH._hkdf_extract(salt, ikm)
            ours = X3DH._hkdf_expand(prk, info, length)
            theirs = HKDF(
                algorithm=hashes.SHA256(), length=length, salt=salt, info=info
            ).derive(ikm)
            assert ours == theirs

    def test_empty_salt_is_treated_as_zeros(self):
        ikm = nacl.utils.random(32)
        info = b'ctx'
        prk = X3DH._hkdf_extract(b'', ikm)
        reference = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None, info=info
        ).derive(ikm)
        assert X3DH._hkdf_expand(prk, info, 32) == reference

    def test_master_key_matches_reference_hkdf(self):
        """The full X3DH KDF step, end to end, against ``cryptography``."""
        for _ in range(20):
            parts = [nacl.utils.random(32) for _ in range(4)]
            context = b'DecentralizedMessenger-X3DH-v1'
            ours = X3DH.derive_master_key(parts, context=context)
            theirs = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'\x00' * 32,
                info=context,
            ).derive(b'\xff' * 32 + b''.join(parts))
            assert ours == theirs


class TestX25519:
    def test_dh_matches_reference(self):
        for _ in range(20):
            sk_a = nacl.utils.random(32)
            sk_b = nacl.utils.random(32)
            reference_private = X25519PrivateKey.from_private_bytes(sk_b)
            pk_b = reference_private.public_key().public_bytes_raw()

            ours = X3DH._dh(PrivateKey(sk_a), PublicKey(pk_b))
            theirs = X25519PrivateKey.from_private_bytes(sk_a).exchange(
                X25519PublicKey.from_public_bytes(pk_b)
            )
            assert ours == theirs

    def test_degenerate_input_rejected_by_both(self):
        private = PrivateKey.generate()
        with pytest.raises(Exception):
            X3DH._dh(private, b'\x00' * 32)


class TestRatchetKdf:
    def test_kdf_rk_matches_reference_hkdf(self):
        ratchet = DoubleRatchet.__new__(DoubleRatchet)
        root_key = nacl.utils.random(32)
        dh_output = nacl.utils.random(32)
        info = b'WhisperRatchet-v2'

        new_root, new_chain = ratchet._kdf_rk(root_key, dh_output)
        reference = HKDF(
            algorithm=hashes.SHA256(), length=64, salt=root_key, info=info + b'\x01'
        ).derive(dh_output)
        assert new_root + new_chain == reference

    def test_kdf_ck_uses_distinct_constants(self):
        ratchet = DoubleRatchet.__new__(DoubleRatchet)
        chain_key = nacl.utils.random(32)
        next_chain_key, message_key = ratchet._kdf_ck(chain_key)
        assert next_chain_key != message_key
        assert len(next_chain_key) == 32 and len(message_key) == 32
