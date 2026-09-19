"""
Isolated X3DH DH test to identify which specific DH operation fails.
This test verifies each DH computation separately before concatenation and HKDF.
"""

from nacl.bindings import (
    crypto_sign_keypair as crypto_sign_ed25519_keypair,
    crypto_sign_ed25519_pk_to_curve25519,
    crypto_sign_ed25519_sk_to_curve25519,
    crypto_scalarmult,
    crypto_box_keypair,  # For X25519 keypairs (SPK, OPK, EK)
)


def test_x3dh_isolated_dh():
    """Test each DH operation in X3DH separately."""
    
    # Generate Identity Keys as Ed25519 (both Alice and Bob)
    IKA_pk, IKA_sk = crypto_sign_ed25519_keypair()
    IKB_pk, IKB_sk = crypto_sign_ed25519_keypair()
    
    # Generate Ephemeral Key as Ed25519 (Alice) - will convert to X25519
    EKA_pk, EKA_sk = crypto_sign_ed25519_keypair()
    
    # Generate Signed PreKey and One-Time PreKey as X25519 directly (Bob)
    # This is the correct approach per Signal spec: SPK/OPK are X25519 keys
    SPKB_pk, SPKB_sk = crypto_box_keypair()
    OPKB_pk, OPKB_sk = crypto_box_keypair()
    
    # Convert Ed25519 keys to X25519 for DH operations
    IKA_pk_x = crypto_sign_ed25519_pk_to_curve25519(IKA_pk)
    IKA_sk_x = crypto_sign_ed25519_sk_to_curve25519(IKA_sk)
    
    EKA_pk_x = crypto_sign_ed25519_pk_to_curve25519(EKA_pk)
    EKA_sk_x = crypto_sign_ed25519_sk_to_curve25519(EKA_sk)
    
    IKB_pk_x = crypto_sign_ed25519_pk_to_curve25519(IKB_pk)
    IKB_sk_x = crypto_sign_ed25519_sk_to_curve25519(IKB_sk)
    
    # SPKB and OPKB are already X25519, no conversion needed
    
    # Alice computes DH values
    # DH1 = DH(IKA_sk, SPKB_pk) - Identity key of Alice × Signed prekey of Bob
    dh1_alice = crypto_scalarmult(IKA_sk_x, SPKB_pk)
    
    # DH2 = DH(EKA_sk, IKB_pk) - Ephemeral key of Alice × Identity key of Bob
    dh2_alice = crypto_scalarmult(EKA_sk_x, IKB_pk_x)
    
    # DH3 = DH(EKA_sk, SPKB_pk) - Ephemeral key of Alice × Signed prekey of Bob
    dh3_alice = crypto_scalarmult(EKA_sk_x, SPKB_pk)
    
    # DH4 = DH(EKA_sk, OPKB_pk) - Ephemeral key of Alice × One-time prekey of Bob
    dh4_alice = crypto_scalarmult(EKA_sk_x, OPKB_pk)
    
    # Bob computes DH values
    # DH1 = DH(SPKB_sk, IKA_pk) - Signed prekey of Bob × Identity key of Alice
    dh1_bob = crypto_scalarmult(SPKB_sk, IKA_pk_x)
    
    # DH2 = DH(IKB_sk, EKA_pk) - Identity key of Bob × Ephemeral key of Alice
    dh2_bob = crypto_scalarmult(IKB_sk_x, EKA_pk_x)
    
    # DH3 = DH(SPKB_sk, EKA_pk) - Signed prekey of Bob × Ephemeral key of Alice
    dh3_bob = crypto_scalarmult(SPKB_sk, EKA_pk_x)
    
    # DH4 = DH(OPKB_sk, EKA_pk) - One-time prekey of Bob × Ephemeral key of Alice
    dh4_bob = crypto_scalarmult(OPKB_sk, EKA_pk_x)
    
    # Verify each DH matches
    print("Testing DH1 (IKA × SPKB)...")
    assert dh1_alice == dh1_bob, "DH1 MISMATCH: Identity key exchange failed"
    print("DH1: OK ✓")
    
    print("Testing DH2 (EKA × IKB)...")
    assert dh2_alice == dh2_bob, "DH2 MISMATCH: Ephemeral × Identity exchange failed"
    print("DH2: OK ✓")
    
    print("Testing DH3 (EKA × SPKB)...")
    assert dh3_alice == dh3_bob, "DH3 MISMATCH: Ephemeral × Signed PreKey exchange failed"
    print("DH3: OK ✓")
    
    print("Testing DH4 (EKA × OPKB)...")
    assert dh4_alice == dh4_bob, "DH4 MISMATCH: Ephemeral × One-time PreKey exchange failed"
    print("DH4: OK ✓")
    
    print("\nAll DH operations match! X3DH primitive is correct.")
    
    return {
        'dh1': dh1_alice,
        'dh2': dh2_alice,
        'dh3': dh3_alice,
        'dh4': dh4_alice,
    }


def test_x3dh_full_kdf():
    """Test complete X3DH with KDF to produce shared secret."""
    
    dh_results = test_x3dh_isolated_dh()
    
    # Concatenate: F || DH1 || DH2 || DH3 || DH4
    # F = 32 bytes of 0xFF for X25519
    F = b'\xff' * 32
    km = F + dh_results['dh1'] + dh_results['dh2'] + dh_results['dh3'] + dh_results['dh4']
    
    print(f"\nKey Material length: {len(km)} bytes (expected 160)")
    assert len(km) == 160, f"Key material should be 160 bytes, got {len(km)}"
    
    # HKDF-SHA256
    import hmac
    import hashlib
    
    def hkdf_extract(salt, ikm):
        """HKDF Extract step using HMAC-SHA256."""
        if not salt:
            salt = b'\x00' * 32
        # HMAC(key=salt, message=ikm)
        return hmac.new(salt, ikm, hashlib.sha256).digest()
    
    def hkdf_expand(prk, info, length):
        """HKDF Expand step using HMAC-SHA256."""
        okm = b''
        t = b''
        i = 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t
            i += 1
        return okm[:length]
    
    # Derive Shared Key (SK)
    salt = b'\x00' * 32  # Zero salt as per X3DH spec
    info = b"MyApp-X3DH-v1"  # Fixed context string
    prk = hkdf_extract(salt, km)
    SK = hkdf_expand(prk, info, 32)
    
    print(f"Shared Key (SK): {SK.hex()}")
    print(f"Shared Key length: {len(SK)} bytes (expected 32)")
    
    return SK


if __name__ == "__main__":
    print("=" * 60)
    print("X3DH Isolated DH Test")
    print("=" * 60)
    test_x3dh_isolated_dh()
    
    print("\n" + "=" * 60)
    print("X3DH Full KDF Test")
    print("=" * 60)
    sk = test_x3dh_full_kdf()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
