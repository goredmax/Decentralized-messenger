"""
Тесты для X3DH implementation

Включают проверку:
1. Базовый handshake с one-time prekey
2. Handshake без one-time prekey
3. Проверка подписи signed prekey
4. Одинаковость shared secret у обеих сторон
"""

import pytest
from src.crypto.x3dh import X3DH, PreKeyBundle


class TestX3DH:
    
    def test_full_handshake_with_one_time_prekey(self):
        """Полный handshake с использованием one-time prekey"""
        x3dh = X3DH()
        
        # Генерация ключей для Alice (инициатор)
        alice_identity_priv, alice_identity_pub = x3dh.generate_identity_keys()
        
        # Генерация ключей для Bob (получатель)
        bob_identity_priv, bob_identity_pub = x3dh.generate_identity_keys()
        bob_spk_priv, bob_spk_pub, bob_spk_signature = x3dh.generate_signed_prekey(bob_identity_priv)
        bob_opk_id, bob_opk_priv, bob_opk_pub = x3dh.generate_one_time_prekey()
        
        # Создаем PreKey bundle для Bob
        bundle = PreKeyBundle(
            identity_key=bob_identity_pub,
            signed_prekey=bob_spk_pub,
            signed_prekey_signature=bob_spk_signature,
            one_time_prekey=bob_opk_pub,
            one_time_prekey_id=bob_opk_id
        )
        
        # Alice инициирует handshake
        alice_state, alice_ephemeral_pub = x3dh.initiate_handshake(
            alice_identity_priv,
            bundle
        )
        
        # Bob обрабатывает handshake
        # Конвертируем Alice identity public key в X25519 формат
        alice_identity_x_pub = alice_identity_priv.verify_key.to_curve25519_public_key()
        
        bob_state = x3dh.receive_handshake(
            bob_identity_priv,
            bob_spk_priv,
            bob_opk_priv,
            alice_ephemeral_pub,
            alice_identity_x_pub  # Передаем уже конвертированный ключ
        )
        
        # Проверяем что shared secret одинаковый
        assert alice_state.shared_secret == bob_state.shared_secret
        assert len(alice_state.shared_secret) == 128  # 4 * 32 bytes
        
    def test_handshake_without_one_time_prekey(self):
        """Handshake без one-time prekey (fallback режим)"""
        x3dh = X3DH()
        
        # Генерация ключей
        alice_identity_priv, alice_identity_pub = x3dh.generate_identity_keys()
        bob_identity_priv, bob_identity_pub = x3dh.generate_identity_keys()
        bob_spk_priv, bob_spk_pub, bob_spk_signature = x3dh.generate_signed_prekey(bob_identity_priv)
        
        # Bundle без one-time prekey
        bundle = PreKeyBundle(
            identity_key=bob_identity_pub,
            signed_prekey=bob_spk_pub,
            signed_prekey_signature=bob_spk_signature,
            one_time_prekey=None,
            one_time_prekey_id=None
        )
        
        # Alice инициирует handshake
        alice_state, alice_ephemeral_pub = x3dh.initiate_handshake(
            alice_identity_priv,
            bundle
        )
        
        # Bob обрабатывает handshake (без OTPK)
        alice_identity_x_pub = alice_identity_priv.verify_key.to_curve25519_public_key()
        
        bob_state = x3dh.receive_handshake(
            bob_identity_priv,
            bob_spk_priv,
            None,  # Нет one-time prekey
            alice_ephemeral_pub,
            alice_identity_x_pub
        )
        
        # Проверяем что shared secret одинаковый
        assert alice_state.shared_secret == bob_state.shared_secret
        assert len(alice_state.shared_secret) == 96  # 3 * 32 bytes
        
    def test_derive_master_key(self):
        """Проверка derive master key через HKDF"""
        x3dh = X3DH()
        
        # Генерируем произвольный shared secret
        shared_secret = b'\x00' * 96
        
        # Derive master key
        master_key = x3dh.derive_master_key(shared_secret, info=b"test")
        
        # Проверяем длину и детерминированность
        assert len(master_key) == 32
        
        # Повторный вызов должен дать тот же результат
        master_key2 = x3dh.derive_master_key(shared_secret, info=b"test")
        assert master_key == master_key2
        
        # Разный info должен дать разный ключ
        master_key3 = x3dh.derive_master_key(shared_secret, info=b"different")
        assert master_key != master_key3
        
    def test_multiple_handshakes_different_secrets(self):
        """Каждый handshake должен давать уникальный shared secret"""
        x3dh = X3DH()
        
        # Ключи Alice
        alice_identity_priv, alice_identity_pub = x3dh.generate_identity_keys()
        
        # Ключи Bob
        bob_identity_priv, bob_identity_pub = x3dh.generate_identity_keys()
        bob_spk_priv, bob_spk_pub, bob_spk_signature = x3dh.generate_signed_prekey(bob_identity_priv)
        
        secrets = []
        
        # Выполняем несколько handshakes
        for _ in range(5):
            # Новый one-time prekey для каждого handshake
            _, bob_opk_priv, bob_opk_pub = x3dh.generate_one_time_prekey()
            
            bundle = PreKeyBundle(
                identity_key=bob_identity_pub,
                signed_prekey=bob_spk_pub,
                signed_prekey_signature=bob_spk_signature,
                one_time_prekey=bob_opk_pub
            )
            
            alice_state, alice_ephemeral_pub = x3dh.initiate_handshake(
                alice_identity_priv,
                bundle
            )
            
            secrets.append(alice_state.shared_secret)
        
        # Все секреты должны быть уникальными
        assert len(set(secrets)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
