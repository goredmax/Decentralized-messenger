"""
Double Ratchet Protocol implementation based on Signal Protocol specification.
Provides forward secrecy and post-compromise security for E2EE messaging.

Reference: https://signal.org/docs/specifications/doubleratchet/
"""

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import nacl.public as public
import nacl.bindings
import nacl.utils


# Constants from Signal specification
MAX_SKIP = 1000  # Maximum number of out-of-order messages to cache
HKDF_INFO = b"WhisperRatchet"
NONCE_SIZE = 24  # XChaCha20-Poly1305 nonce size


@dataclass
class SessionState:
    """Complete state of a Double Ratchet session."""
    # DH ratchet keys
    dh_local_priv: bytes  # Our current ephemeral private key
    dh_local_pub: bytes   # Our current ephemeral public key
    dh_remote_pub: Optional[bytes]  # Remote's current ephemeral public key
    
    # Root and chain keys
    root_key: bytes       # 32 bytes
    send_chain_key: Optional[bytes]  # Sending chain key
    recv_chain_key: Optional[bytes]  # Receiving chain key
    
    # Counters
    send_msg_count: int = 0      # Messages sent in current sending chain
    recv_msg_count: int = 0      # Messages received in current receiving chain
    prev_send_count: int = 0     # Number of messages in previous sending chain (PN)
    
    # Skipped message keys: key = (dh_remote_hex, msg_num), value = message_key
    skipped_keys: Dict[Tuple[str, int], bytes] = field(default_factory=dict)
    
    def serialize(self) -> dict:
        """Serialize state to dictionary for storage."""
        return {
            'dh_local_priv': self.dh_local_priv.hex(),
            'dh_local_pub': self.dh_local_pub.hex(),
            'dh_remote_pub': self.dh_remote_pub.hex() if self.dh_remote_pub else None,
            'root_key': self.root_key.hex(),
            'send_chain_key': self.send_chain_key.hex() if self.send_chain_key else None,
            'recv_chain_key': self.recv_chain_key.hex() if self.recv_chain_key else None,
            'send_msg_count': self.send_msg_count,
            'recv_msg_count': self.recv_msg_count,
            'prev_send_count': self.prev_send_count,
            'skipped_keys': {(k[0], k[1]): v.hex() for k, v in self.skipped_keys.items()},
        }
    
    @classmethod
    def deserialize(cls, data: dict) -> 'SessionState':
        """Deserialize state from dictionary."""
        return cls(
            dh_local_priv=bytes.fromhex(data['dh_local_priv']),
            dh_local_pub=bytes.fromhex(data['dh_local_pub']),
            dh_remote_pub=bytes.fromhex(data['dh_remote_pub']) if data['dh_remote_pub'] else None,
            root_key=bytes.fromhex(data['root_key']),
            send_chain_key=bytes.fromhex(data['send_chain_key']) if data['send_chain_key'] else None,
            recv_chain_key=bytes.fromhex(data['recv_chain_key']) if data['recv_chain_key'] else None,
            send_msg_count=data['send_msg_count'],
            recv_msg_count=data['recv_msg_count'],
            prev_send_count=data['prev_send_count'],
            skipped_keys={(k[0], k[1]): bytes.fromhex(v) for k, v in data['skipped_keys'].items()},
        )


class DoubleRatchet:
    """
    Double Ratchet implementation following Signal Protocol specification.
    
    Key points:
    - KDF_RK: HKDF-SHA256 with RK as salt, DH output as IKM
    - KDF_CK: HMAC-SHA256 with 0x01 for message key, 0x02 for next chain key
    - AEAD: XChaCha20-Poly1305 for encryption
    - MAX_SKIP limit for DoS protection
    """
    
    def __init__(self, 
                 dh_private: public.PrivateKey,
                 remote_dh_public: Optional[bytes],
                 root_key: bytes,
                 is_initiator: bool = True):
        """
        Initialize Double Ratchet session.
        
        Args:
            dh_private: Our initial DH private key
            remote_dh_public: Remote's initial DH public key (None for recipient before first msg)
            root_key: Root key from X3DH handshake
            is_initiator: True if we initiated the conversation (Alice)
        """
        self._state = SessionState(
            dh_local_priv=bytes(dh_private),
            dh_local_pub=bytes(dh_private.public_key),
            dh_remote_pub=remote_dh_public,
            root_key=root_key,
            send_chain_key=None,
            recv_chain_key=None,
            send_msg_count=0,
            recv_msg_count=0,
            prev_send_count=0,
        )
        self._dh_private = dh_private
        
        # Initiator (Alice) generates new ephemeral key for first message
        # Recipient (Bob) waits for Alice's first message
        if is_initiator and remote_dh_public is not None:
            self._perform_dh_ratchet_as_sender()
    
    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state
    
    def _dh(self, private: public.PrivateKey, pub: bytes) -> bytes:
        """Perform X25519 Diffie-Hellman."""
        return public.Box(private, public.PublicKey(pub))._shared_key
    
    def _kdf_rk(self, root_key: bytes, dh_output: bytes) -> Tuple[bytes, bytes]:
        """
        KDF for root key update (Signal spec).
        
        HKDF-SHA256(salt=root_key, ikm=dh_output, info="WhisperRatchet")
        Returns: (new_root_key, new_chain_key)
        
        IMPORTANT: root_key goes in salt, dh_output in ikm (not reversed!)
        """
        # HKDF-Extract: PRK = HMAC-SHA256(salt=root_key, msg=dh_output)
        prk = hmac.new(root_key, dh_output, hashlib.sha256).digest()
        
        # HKDF-Expand: OKM = HMAC-SHA256(PRK, info || 0x01)
        okm = hmac.new(prk, HKDF_INFO + b'\x01', hashlib.sha256).digest()
        
        # Split into two 32-byte keys
        return okm[:32], okm[32:64]
    
    def _kdf_ck(self, chain_key: bytes) -> Tuple[bytes, bytes]:
        """
        KDF for chain key update (Signal spec).
        
        Returns: (next_chain_key, message_key)
        
        IMPORTANT: 0x01 for message key, 0x02 for next chain key (not reversed!)
        """
        message_key = hmac.new(chain_key, b'\x01', hashlib.sha256).digest()
        next_chain_key = hmac.new(chain_key, b'\x02', hashlib.sha256).digest()
        return next_chain_key, message_key
    
    def _perform_dh_ratchet_as_sender(self):
        """Generate new ephemeral key and derive sending chain."""
        # Generate new ephemeral keypair
        new_dh_private = public.PrivateKey.generate()
        self._dh_private = new_dh_private
        self._state.dh_local_priv = bytes(new_dh_private)
        self._state.dh_local_pub = bytes(new_dh_private.public_key)
        
        # DH with remote's public key
        if self._state.dh_remote_pub is None:
            raise ValueError("Remote DH public key required for DH ratchet")
        
        dh_output = self._dh(new_dh_private, self._state.dh_remote_pub)
        
        # Derive new root key and sending chain key
        self._state.root_key, self._state.send_chain_key = self._kdf_rk(
            self._state.root_key, dh_output
        )
        self._state.send_msg_count = 0
    
    def _perform_dh_ratchet_as_receiver(self, new_remote_dh: bytes):
        """
        Perform DH ratchet when receiving a message with new DH key.
        
        This is called when header.dh != current dh_remote_pub.
        """
        # Store previous send count (PN) before switching chains
        self._state.prev_send_count = self._state.send_msg_count
        self._state.send_msg_count = 0
        self._state.recv_msg_count = 0
        
        # First DH ratchet: derive receiving chain from old root key
        if self._state.dh_remote_pub is not None:
            # We had a previous remote key, skip those messages first
            self._skip_message_keys(self._state.prev_send_count)
        
        # DH with sender's new public key using our current private key
        dh_output = self._dh(self._dh_private, new_remote_dh)
        
        # Update root key and derive receiving chain
        self._state.root_key, self._state.recv_chain_key = self._kdf_rk(
            self._state.root_key, dh_output
        )
        
        # Update remote DH public key
        self._state.dh_remote_pub = new_remote_dh
        
        # Second DH ratchet: generate new sending key and derive sending chain
        new_dh_private = public.PrivateKey.generate()
        self._dh_private = new_dh_private
        self._state.dh_local_priv = bytes(new_dh_private)
        self._state.dh_local_pub = bytes(new_dh_private.public_key)
        
        dh_output_2 = self._dh(new_dh_private, new_remote_dh)
        self._state.root_key, self._state.send_chain_key = self._kdf_rk(
            self._state.root_key, dh_output_2
        )
    
    def _skip_message_keys(self, until: int):
        """
        Skip message keys up to 'until' counter value.
        Stores them in skipped_keys for later out-of-order decryption.
        
        Enforces MAX_SKIP limit for DoS protection.
        """
        if self._state.recv_chain_key is None:
            return
        
        if self._state.recv_msg_count + MAX_SKIP < until:
            raise TooManySkippedMessagesError(
                f"Skipping {until - self._state.recv_msg_count} messages exceeds MAX_SKIP={MAX_SKIP}"
            )
        
        ck = self._state.recv_chain_key
        while self._state.recv_msg_count < until:
            ck, mk = self._kdf_ck(ck)
            key_id = (bytes(self._state.dh_remote_pub).hex(), self._state.recv_msg_count)
            self._state.skipped_keys[key_id] = mk
            self._state.recv_msg_count += 1
        
        self._state.recv_chain_key = ck
    
    def encrypt(self, plaintext: bytes, associated_data: bytes = b'') -> Tuple[bytes, dict]:
        """
        Encrypt a message using Double Ratchet.
        
        Returns: (ciphertext_with_nonce, header_dict)
        
        Header format (sent in clear, authenticated via AEAD):
        {
            'dh': hex-encoded DH public key,
            'pn': previous send count (messages in previous chain),
            'n': current message number in this chain
        }
        """
        # Ensure we have a sending chain
        if self._state.send_chain_key is None:
            self._perform_dh_ratchet_as_sender()
        
        # Derive message key and advance chain
        self._state.send_chain_key, message_key = self._kdf_ck(self._state.send_chain_key)
        
        # Build header (sent in clear)
        header = {
            'dh': self._state.dh_local_pub.hex(),
            'pn': self._state.prev_send_count,
            'n': self._state.send_msg_count,
        }
        
        # Serialize header for AD
        header_bytes = self._serialize_header(header)
        ad = associated_data + header_bytes
        
        # Generate random 24-byte nonce for XChaCha20
        nonce = nacl.utils.random(NONCE_SIZE)
        
        # Encrypt using XChaCha20-Poly1305
        ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            plaintext, ad, nonce, message_key
        )
        
        # Prepend nonce to ciphertext
        encrypted = nonce + ciphertext
        
        self._state.send_msg_count += 1
        
        return encrypted, header
    
    def decrypt(self, encrypted: bytes, header: dict, associated_data: bytes = b'') -> bytes:
        """
        Decrypt a message using Double Ratchet.
        
        Algorithm (strict order from Signal spec):
        1. Check if message key is in skipped_keys
        2. If header.dh != dh_remote_pub: perform DH ratchet
        3. Skip message keys up to header.n
        4. Derive message key and decrypt
        5. Remove used key from skipped_keys if present
        
        Returns: plaintext
        """
        dh_hex = header['dh']
        msg_num = header['n']
        
        # Serialize header for AD verification
        header_bytes = self._serialize_header(header)
        ad = associated_data + header_bytes
        
        nonce = encrypted[:NONCE_SIZE]
        ciphertext = encrypted[NONCE_SIZE:]
        
        # Step 1: Check skipped keys first
        key_id = (dh_hex, msg_num)
        if key_id in self._state.skipped_keys:
            message_key = self._state.skipped_keys.pop(key_id)
            
            # Limit skipped keys size
            if len(self._state.skipped_keys) > MAX_SKIP:
                # Remove oldest entry
                oldest_key = min(self._state.skipped_keys.keys())
                del self._state.skipped_keys[oldest_key]
            
            # Decrypt with skipped key
            return self._do_decrypt(message_key, nonce, ciphertext, ad)
        
        # Step 2: Check if DH key changed (DH ratchet needed)
        current_dh_hex = self._state.dh_remote_pub.hex() if self._state.dh_remote_pub else None
        if dh_hex != current_dh_hex:
            # Perform DH ratchet with sender's new DH public key
            new_remote_dh = bytes.fromhex(dh_hex)
            self._perform_dh_ratchet_as_receiver(new_remote_dh)
        
        # Step 3: Skip message keys up to header.n
        if msg_num < self._state.recv_msg_count:
            raise DuplicateMessageError(f"Message {msg_num} already received (current: {self._state.recv_msg_count})")
        
        self._skip_message_keys(msg_num)
        
        # Step 4: Derive message key for this message
        if self._state.recv_chain_key is None:
            raise InvalidSessionStateError("Receive chain key is None")
        
        self._state.recv_chain_key, message_key = self._kdf_ck(self._state.recv_chain_key)
        self._state.recv_msg_count += 1
        
        # Step 5: Decrypt
        return self._do_decrypt(message_key, nonce, ciphertext, ad)
    
    def _do_decrypt(self, message_key: bytes, nonce: bytes, ciphertext: bytes, ad: bytes) -> bytes:
        """Perform actual AEAD decryption."""
        try:
            plaintext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
                ciphertext, ad, nonce, message_key
            )
        except Exception:
            raise DecryptionError("AEAD decryption failed - message corrupted or tampered")
        
        return plaintext
    
    def _serialize_header(self, header: dict) -> bytes:
        """Serialize header to bytes for AD."""
        # Simple deterministic serialization: dh||pn||n
        dh = bytes.fromhex(header['dh'])
        pn_bytes = header['pn'].to_bytes(4, 'big')
        n_bytes = header['n'].to_bytes(4, 'big')
        return dh + pn_bytes + n_bytes


# Custom exceptions
class TooManySkippedMessagesError(Exception):
    """Raised when trying to skip more than MAX_SKIP messages (DoS protection)."""
    pass


class DuplicateMessageError(Exception):
    """Raised when receiving a duplicate message."""
    pass


class InvalidSessionStateError(Exception):
    """Raised when session state is invalid for the operation."""
    pass


class DecryptionError(Exception):
    """Raised when AEAD decryption fails."""
    pass
