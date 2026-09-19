"""
Key management module for E2EE.
Uses X25519 for key exchange and Ed25519 for signatures.
Based on libsodium (PyNaCl) - a proven, audited cryptographic library.
"""

import nacl.public as public
from nacl.public import PrivateKey, PublicKey
import nacl.signing as signing
from nacl.utils import random
from typing import Tuple


class KeyPair:
    """
    Manages asymmetric key pairs for encryption and signing.
    
    Uses:
    - X25519 for ECDH key exchange (encryption)
    - Ed25519 for digital signatures (authentication)
    """
    
    def __init__(self):
        # Generate X25519 keypair for encryption
        self._encrypt_private = PrivateKey.generate()
        self._encrypt_public = self._encrypt_private.public_key
        
        # Generate Ed25519 keypair for signing
        self._sign_private = signing.SigningKey.generate()
        self._sign_public = self._sign_private.verify_key
    
    @property
    def encrypt_private_key(self) -> bytes:
        """Return private key for encryption (keep secret!)"""
        return bytes(self._encrypt_private)
    
    @property
    def encrypt_public_key(self) -> bytes:
        """Return public key for encryption (share with others)"""
        return bytes(self._encrypt_public)
    
    @property
    def sign_private_key(self) -> bytes:
        """Return private key for signing (keep secret!)"""
        return bytes(self._sign_private)
    
    @property
    def sign_public_key(self) -> bytes:
        """Return public key for signing (share with others)"""
        return bytes(self._sign_public)
    
    def export_public_keys(self) -> dict:
        """Export public keys for sharing with other users."""
        return {
            'encrypt': self.encrypt_public_key.hex(),
            'sign': self.sign_public_key.hex()
        }
    
    @classmethod
    def import_public_keys(cls, keys: dict) -> 'RemotePublicKey':
        """Import another user's public keys."""
        return RemotePublicKey(
            encrypt_key=bytes.fromhex(keys['encrypt']),
            sign_key=bytes.fromhex(keys['sign'])
        )


class RemotePublicKey:
    """
    Stores remote user's public keys for secure communication.
    """
    
    def __init__(self, encrypt_key: bytes, sign_key: bytes):
        self._encrypt_public = public.PublicKey(encrypt_key)
        self._sign_public = signing.VerifyKey(sign_key)
    
    @property
    def encrypt_public_key(self) -> public.PublicKey:
        return self._encrypt_public
    
    @property
    def sign_public_key(self) -> signing.VerifyKey:
        return self._sign_public


class Box:
    """
    Implements authenticated encryption using NaCl box.
    Combines X25519 key exchange with AES-256-GCM encryption.
    Provides forward secrecy when used with Double Ratchet.
    """
    
    def __init__(self, private_key: PrivateKey, public_key: PublicKey):
        """
        Create encrypted channel between two parties.
        
        Args:
            private_key: Your private encryption key
            public_key: Recipient's public encryption key
        """
        self._box = public.Box(private_key, public_key)
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt message with authentication.
        
        Returns:
            Nonce + ciphertext (safe to transmit)
        """
        return self._box.encrypt(plaintext)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt and authenticate message.
        
        Raises:
            CryptoError if authentication fails
        """
        return self._box.decrypt(ciphertext)


class Signer:
    """
    Digital signature operations using Ed25519.
    Used for message authentication and identity verification.
    """
    
    def __init__(self, private_key: signing.SigningKey):
        self._signer = private_key
    
    def sign(self, message: bytes) -> bytes:
        """Sign a message. Returns signed message."""
        return self._signer.sign(message)
    
    def verify(self, signed_message: bytes, verifier: signing.VerifyKey) -> bytes:
        """
        Verify signature and return original message.
        
        Raises:
            BadSignature if verification fails
        """
        return verifier.verify(signed_message)


def generate_shared_secret(private_key: PrivateKey, 
                          public_key: PublicKey) -> bytes:
    """
    Generate shared secret using X25519 ECDH.
    Used as basis for Double Ratchet protocol.
    """
    return public.Box(private_key, public_key)._shared_key


# Example usage
if __name__ == "__main__":
    # Alice generates keys
    alice = KeyPair()
    print("Alice's public keys:", alice.export_public_keys())
    
    # Bob generates keys
    bob = KeyPair()
    print("Bob's public keys:", bob.export_public_keys())
    
    # Alice creates box to send to Bob
    bob_keys = KeyPair.import_public_keys(bob.export_public_keys())
    alice_to_bob = Box(alice._encrypt_private, bob_keys.encrypt_public_key)
    
    # Bob creates box to receive from Alice
    alice_keys = KeyPair.import_public_keys(alice.export_public_keys())
    bob_from_alice = Box(bob._encrypt_private, alice_keys.encrypt_public_key)
    
    # Test encryption/decryption
    message = b"Hello, secure world!"
    encrypted = alice_to_bob.encrypt(message)
    decrypted = bob_from_alice.decrypt(encrypted)
    
    assert decrypted == message
    print("✓ E2EE test passed!")
    
    # Test signing
    alice_signer = Signer(alice._sign_private)
    signed = alice_signer.sign(message)
    
    verified = alice_signer.verify(signed, alice._sign_public)
    assert verified == message
    print("✓ Signature test passed!")
