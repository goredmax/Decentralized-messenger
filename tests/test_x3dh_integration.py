"""
X3DH Integration Test - Full Protocol Verification

Проверяет что обе стороны (Alice и Bob) получают идентичный shared secret
после полного X3DH handshake.
"""

import pytest
from src.crypto.x3dh import X3DH, PreKeyBundle


def test_x3dh_full_handshake_with_otk():
    """Полный X3DH handshake с one-time prekey."""
    x3dh = X3DH()
    
    # === Инициализация Alice ===
    alice_ik, alice_ik_pub = x3dh.generate_identity_keys()
    
    # === Инициализация Bob ===
    bob_ik, bob_ik_pub = x3dh.generate_identity_keys()
    bob_spk_priv, bob_spk_pub, bob_spk_sig = x3dh.generate_signed_prekey(bob_ik)
    bob_otk_id, bob_otk_priv, bob_otk_pub = x3dh.generate_one_time_prekey()
    
    # Bob публикует prekey bundle
    bob_bundle = PreKeyBundle(
        identity_key=bob_ik_pub,
        signed_prekey=bob_spk_pub,
        signed_prekey_signature=bob_spk_sig,
        one_time_prekey=bob_otk_pub,
        one_time_prekey_id=bob_otk_id,
    )
    
    # === Alice инициирует handshake ===
    alice_state, alice_ephemeral_pub = x3dh.initiate_handshake(alice_ik, bob_bundle)
    
    # === Bob обрабатывает handshake ===
    # Конвертируем identity key Alice в X25519 для Bob
    alice_ik_x25519 = alice_ik.verify_key.to_curve25519_public_key()
    
    bob_state = x3dh.receive_handshake(
        identity_private=bob_ik,
        signed_prekey_private=bob_spk_priv,
        one_time_prekey_private=bob_otk_priv,
        ephemeral_public=alice_ephemeral_pub,
        initiator_identity_x25519=alice_ik_x25519,
    )
    
    # === Обе стороны выводят master key ===
    alice_master_key = x3dh.derive_master_key(alice_state.shared_secret)
    bob_master_key = x3dh.derive_master_key(bob_state.shared_secret)
    
    # === Проверка: ключи должны совпадать ===
    assert alice_master_key == bob_master_key, (
        f"Master keys don't match!\n"
        f"Alice: {alice_master_key.hex()}\n"
        f"Bob:   {bob_master_key.hex()}"
    )
    
    print(f"✓ Master key совпадает: {alice_master_key.hex()}")
    return alice_master_key


def test_x3dh_handshake_without_otk():
    """X3DH handshake без one-time prekey (OPK опционален)."""
    x3dh = X3DH()
    
    # === Инициализация Alice ===
    alice_ik, alice_ik_pub = x3dh.generate_identity_keys()
    
    # === Инициализация Bob ===
    bob_ik, bob_ik_pub = x3dh.generate_identity_keys()
    bob_spk_priv, bob_spk_pub, bob_spk_sig = x3dh.generate_signed_prekey(bob_ik)
    # Не генерируем one-time prekey
    
    # Bob публикует prekey bundle БЕЗ one-time prekey
    bob_bundle = PreKeyBundle(
        identity_key=bob_ik_pub,
        signed_prekey=bob_spk_pub,
        signed_prekey_signature=bob_spk_sig,
        one_time_prekey=None,
        one_time_prekey_id=None,
    )
    
    # === Alice инициирует handshake ===
    alice_state, alice_ephemeral_pub = x3dh.initiate_handshake(alice_ik, bob_bundle)
    
    # === Bob обрабатывает handshake ===
    alice_ik_x25519 = alice_ik.verify_key.to_curve25519_public_key()
    
    bob_state = x3dh.receive_handshake(
        identity_private=bob_ik,
        signed_prekey_private=bob_spk_priv,
        one_time_prekey_private=None,  # Нет OTP
        ephemeral_public=alice_ephemeral_pub,
        initiator_identity_x25519=alice_ik_x25519,
    )
    
    # === Обе стороны выводят master key ===
    alice_master_key = x3dh.derive_master_key(alice_state.shared_secret)
    bob_master_key = x3dh.derive_master_key(bob_state.shared_secret)
    
    # === Проверка: ключи должны совпадать ===
    assert alice_master_key == bob_master_key, (
        f"Master keys don't match (no OTK)!\n"
        f"Alice: {alice_master_key.hex()}\n"
        f"Bob:   {bob_master_key.hex()}"
    )
    
    print(f"✓ Master key совпадает (без OTK): {alice_master_key.hex()}")
    return alice_master_key


def test_x3dh_dh_components():
    """Проверка отдельных DH компонентов."""
    x3dh = X3DH()
    
    # Генерируем все ключи
    alice_ik, alice_ik_pub = x3dh.generate_identity_keys()
    bob_ik, bob_ik_pub = x3dh.generate_identity_keys()
    bob_spk_priv, bob_spk_pub, bob_spk_sig = x3dh.generate_signed_prekey(bob_ik)
    bob_otk_id, bob_otk_priv, bob_otk_pub = x3dh.generate_one_time_prekey()
    
    # Конвертируем для DH операций
    alice_ik_x = alice_ik.to_curve25519_private_key()
    bob_ik_x = bob_ik.to_curve25519_private_key()
    
    # Проверяем что DH операции симметричны
    dh1_alice = x3dh._dh(alice_ik_x, bob_spk_pub)
    dh1_bob = x3dh._dh(bob_spk_priv, alice_ik_x.public_key)
    assert dh1_alice == dh1_bob, "DH1 mismatch"
    print("✓ DH1 (IK_A × SPK_B) совпадает")
    
    # Для DH2 нужно ephemeral key
    eph_priv, eph_pub = x3dh.generate_identity_keys()
    eph_x = eph_priv.to_curve25519_private_key()
    
    dh2_alice = x3dh._dh(eph_x, bob_ik_x.public_key)
    dh2_bob = x3dh._dh(bob_ik_x, eph_x.public_key)
    assert dh2_alice == dh2_bob, "DH2 mismatch"
    print("✓ DH2 (EK_A × IK_B) совпадает")
    
    dh3_alice = x3dh._dh(eph_x, bob_spk_pub)
    dh3_bob = x3dh._dh(bob_spk_priv, eph_x.public_key)
    assert dh3_alice == dh3_bob, "DH3 mismatch"
    print("✓ DH3 (EK_A × SPK_B) совпадает")
    
    dh4_alice = x3dh._dh(eph_x, bob_otk_pub)
    dh4_bob = x3dh._dh(bob_otk_priv, eph_x.public_key)
    assert dh4_alice == dh4_bob, "DH4 mismatch"
    print("✓ DH4 (EK_A × OPK_B) совпадает")


if __name__ == "__main__":
    print("=" * 60)
    print("X3DH Integration Tests")
    print("=" * 60)
    
    print("\n1. Testing DH components...")
    test_x3dh_dh_components()
    
    print("\n2. Testing full handshake with OTK...")
    test_x3dh_full_handshake_with_otk()
    
    print("\n3. Testing handshake without OTK...")
    test_x3dh_handshake_without_otk()
    
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
