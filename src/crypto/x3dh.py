"""
X3DH (Extended Triple Diffie-Hellman) Implementation

Спецификация: https://signal.org/docs/specifications/x3dh/

Этот модуль реализует протокол X3DH для установления безопасного сеанса между двумя участниками.
"""

import os
from typing import Tuple, Optional
from dataclasses import dataclass
import nacl
from nacl.public import PrivateKey, PublicKey
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import CryptoError
import hashlib
import hmac


@dataclass
class PreKeyBundle:
    """Набор预ключей для X3DH handshake"""
    identity_key: PublicKey  # IK_B
    signed_prekey: PublicKey  # SPK_B
    signed_prekey_signature: bytes  # Signature(SP K_B)
    one_time_prekey: Optional[PublicKey] = None  # OPK_B (опционально)
    one_time_prekey_id: Optional[int] = None


@dataclass
class X3DHState:
    """Состояние X3DH протокола"""
    shared_secret: bytes
    ephemeral_key: Optional[PrivateKey] = None


class X3DH:
    """
    Extended Triple Diffie-Hellman Key Agreement Protocol
    
    Реализует четыре варианта DH вычислений в зависимости от доступности one-time prekey.
    """
    
    @staticmethod
    def generate_identity_keys() -> Tuple[SigningKey, VerifyKey]:
        """Генерация пары ключей идентичности (долгосрочные)"""
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key
        return signing_key, verify_key
    
    @staticmethod
    def generate_signed_prekey(identity_key: SigningKey) -> Tuple[PrivateKey, PublicKey, bytes]:
        """
        Генерация подписанного预ключа
        
        Возвращает: (spk_private, spk_public, signature)
        """
        spk_private = PrivateKey.generate()
        spk_public = spk_private.public_key
        
        # Подписываем prekey долгосрочным ключом идентичности
        signature = identity_key.sign(spk_public.encode()).signature
        
        return spk_private, spk_public, signature
    
    @staticmethod
    def generate_one_time_prekey() -> Tuple[int, PrivateKey, PublicKey]:
        """
        Генерация одноразового预ключа
        
        Возвращает: (prekey_id, opk_private, opk_public)
        """
        prekey_id = int.from_bytes(os.urandom(4), 'big')
        opk_private = PrivateKey.generate()
        opk_public = opk_private.public_key
        
        return prekey_id, opk_private, opk_public
    
    def _dh(self, private_key: PrivateKey, public_key: PublicKey) -> bytes:
        """
        Базовая DH операция используя X25519
        
        Возвращает 32-byte shared secret
        """
        # Используем низкоуровневую операцию скалярного умножения
        return nacl.bindings.crypto_scalarmult(
            private_key.encode(),
            public_key.encode()
        )
    
    def _dh1(self, private_key: PrivateKey, public_key: PublicKey) -> bytes:
        """DH(IK_A, SPK_B)"""
        return self._dh(private_key, public_key)
    
    def _dh2(self, private_key: PrivateKey, public_key: PublicKey) -> bytes:
        """DH(EK_A, IK_B)"""
        return self._dh(private_key, public_key)
    
    def _dh3(self, private_key: PrivateKey, public_key: PublicKey) -> bytes:
        """DH(EK_A, SPK_B)"""
        return self._dh(private_key, public_key)
    
    def _dh4(self, private_key: PrivateKey, public_key: PublicKey) -> bytes:
        """DH(EK_A, OPK_B)"""
        return self._dh(private_key, public_key)
    
    def initiate_handshake(
        self,
        identity_private: SigningKey,
        bundle: PreKeyBundle
    ) -> Tuple[X3DHState, PublicKey]:
        """
        Инициация handshake (сторона A)
        
        Согласно спецификации Signal X3DH:
        DH1 = DH(IK_A, SPK_B)
        DH2 = DH(EK_A, IK_B)  ← Эфемерный ключ, НЕ Identity!
        DH3 = DH(EK_A, SPK_B)
        DH4 = DH(EK_A, OPK_B) [если есть OPK]
        
        SK = KDF(DH1 || DH2 || DH3 || DH4)
        
        Args:
            identity_private: Долгосрочный ключ идентичности A (SigningKey)
            bundle: PreKey bundle от B
            
        Returns:
            (state, ephemeral_public_key)
        """
        # Для DH операций нам нужен private key в формате X25519
        # Identity key - это Ed25519 signing key, конвертируем в X25519
        identity_private_x = PrivateKey(identity_private.to_curve25519_private_key().encode())
        
        # Генерируем эфемерный ключ
        ephemeral = PrivateKey.generate()
        ephemeral_public = ephemeral.public_key
        
        # Конвертируем Identity public key Bob из Ed25519 в X25519 для DH2
        # bundle.identity_key - это Ed25519 VerifyKey, нужно конвертировать
        bob_ik_x25519 = PublicKey(bundle.identity_key.to_curve25519_public_key().encode())
        
        # Вычисляем DH составляющие согласно спецификации
        # DH1 = DH(IK_A, SPK_B)
        dh1 = self._dh(identity_private_x, bundle.signed_prekey)
        
        # DH2 = DH(EK_A, IK_B) ← ИСПРАВЛЕНО: используем ephemeral и конвертированный IK_B
        dh2 = self._dh(ephemeral, bob_ik_x25519)
        
        # DH3 = DH(EK_A, SPK_B)
        dh3 = self._dh(ephemeral, bundle.signed_prekey)
        
        # Собираем shared secret
        if bundle.one_time_prekey is not None:
            # DH4 = DH(EK_A, OPK_B)
            dh4 = self._dh(ephemeral, bundle.one_time_prekey)
            shared_secret = dh1 + dh2 + dh3 + dh4
        else:
            # Без one-time prekey
            shared_secret = dh1 + dh2 + dh3
        
        state = X3DHState(shared_secret=shared_secret, ephemeral_key=ephemeral)
        return state, ephemeral_public
    
    def receive_handshake(
        self,
        identity_private: SigningKey,
        signed_prekey_private: PrivateKey,
        one_time_prekey_private: Optional[PrivateKey],
        ephemeral_public: PublicKey,
        initiator_identity_x25519: PublicKey  # Уже в X25519 формате
    ) -> X3DHState:
        """
        Обработка handshake (сторона B)
        
        Согласно спецификации Signal X3DH:
        DH1 = DH(A_IK, B_SPK) == DH(B_SPK, A_IK)
        DH2 = DH(A_EK, B_IK)  == DH(B_IK, A_EK) ← Эфемерный ключ Alice!
        DH3 = DH(A_EK, B_SPK) == DH(B_SPK, A_EK)
        DH4 = DH(A_EK, B_OPK) == DH(B_OPK, A_EK) [если есть OPK]
        
        SK = KDF(DH1 || DH2 || DH3 || DH4)
        
        Args:
            identity_private: Долгосрочный ключ идентичности B (SigningKey Ed25519)
            signed_prekey_private: Приватный ключ signed prekey (X25519)
            one_time_prekey_private: Приватный ключ one-time prekey (если есть, X25519)
            ephemeral_public: Эфемерный публичный ключ от A (X25519)
            initiator_identity_x25519: Публичный ключ идентичности A в X25519 формате
            
        Returns:
            X3DHState с shared secret
        """
        # Конвертируем identity key B в X25519 формат
        identity_private_x = PrivateKey(identity_private.to_curve25519_private_key().encode())
        
        # Вычисляем DH составляющие согласно спецификации (симметрично стороне A)
        # DH1 = DH(B_SPK, A_IK)
        dh1 = self._dh(signed_prekey_private, initiator_identity_x25519)
        
        # DH2 = DH(B_IK, A_EK) ← ИСПРАВЛЕНО: используем ephemeral public key Alice
        dh2 = self._dh(identity_private_x, ephemeral_public)
        
        # DH3 = DH(B_SPK, A_EK)
        dh3 = self._dh(signed_prekey_private, ephemeral_public)
        
        # Собираем shared secret
        if one_time_prekey_private is not None:
            # DH4 = DH(B_OPK, A_EK)
            dh4 = self._dh(one_time_prekey_private, ephemeral_public)
            shared_secret = dh1 + dh2 + dh3 + dh4
        else:
            shared_secret = dh1 + dh2 + dh3
        
        return X3DHState(shared_secret=shared_secret)
    
    @staticmethod
    def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
        """HKDF Extract step using HMAC-SHA256."""
        if not salt:
            salt = b'\x00' * 32
        return hmac.new(salt, ikm, hashlib.sha256).digest()
    
    @staticmethod
    def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
        """HKDF Expand step using HMAC-SHA256."""
        okm = b''
        t = b''
        i = 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t
            i += 1
        return okm[:length]
    
    @staticmethod
    def derive_master_key(shared_secret: bytes, info: bytes = b"MyApp-X3DH-v1") -> bytes:
        """
        Derive master key из shared secret используя KDF
        
        Использует HKDF-SHA256 как указано в спецификации Signal X3DH.
        
        Согласно спецификации:
        - DH1 = DH(IK_A, SPK_B) или DH(SP K_B, IK_A)
        - DH2 = DH(EK_A, IK_B) или DH(IK_B, EK_A)
        - DH3 = DH(EK_A, SPK_B) или DH(SP K_B, EK_A)
        - DH4 = DH(EK_A, OPK_B) или DH(OPK_B, EK_A) [если есть OPK]
        
        SK = HKDF-SHA256(
            salt = 32 нулевых байта,
            ikm = F || DH1 || DH2 || DH3 || DH4 (или без DH4),
            info = context string,
            length = 32
        )
        
        где F = 0xFF * 32 для X25519
        
        Args:
            shared_secret: Конкатенация F || DH1 || DH2 || DH3 || DH4
            info: Context string для HKDF (должна быть одинаковой у обеих сторон)
            
        Returns:
            32-byte master key
        """
        # Добавляем префикс F = 0xFF * 32 если его нет
        # Это требуется спецификацией X3DH для X25519
        if len(shared_secret) == 128:  # Только DH1||DH2||DH3||DH4
            F = b'\xff' * 32
            km = F + shared_secret
        elif len(shared_secret) == 96:  # Только DH1||DH2||DH3 (без OPK)
            F = b'\xff' * 32
            km = F + shared_secret
        else:
            # Предполагаем что F уже добавлен
            km = shared_secret
        
        # HKDF-SHA256 с нулевым salt
        salt = b'\x00' * 32
        prk = X3DH._hkdf_extract(salt, km)
        return X3DH._hkdf_expand(prk, info, 32)
