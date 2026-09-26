"""
Double Ratchet, following the Signal specification.

Reference: https://signal.org/docs/specifications/doubleratchet/

Security notes
--------------
``KDF_CK`` below follows the specification: the *next* chain key is
``HMAC(ck, 0x01)`` and the *message* key is ``HMAC(ck, 0x02)``. Getting this
pair backwards is self-consistent and does not break a session on its own, but
it is incompatible with libsignal, so it is fixed here together with an explicit
protocol version bump. Sessions written by earlier versions are rejected on
load rather than silently misinterpreted.

Message keys are deliberately excluded from :meth:`SessionState.serialize` by
default. They are symmetric secrets: storing them next to the session state
turns any storage compromise into a decryption capability for the messages that
are still outstanding.

Decryption operates on a snapshot and rolls back on any failure. Without that,
a single tampered packet advances the chain and desynchronises the session
permanently, and a forged header carrying an attacker-chosen DH key mutates the
root key before any authentication happens. See :meth:`DoubleRatchet.decrypt`.
"""

import hashlib
import hmac
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import nacl.bindings
import nacl.public as public
import nacl.utils

from .kdf import hkdf

# The Signal specification caps how far a receiver will skip ahead. Bounded so
# a peer cannot make us grind through unbounded chain-key derivations.
MAX_SKIP = 1000

# Hard ceiling on total stored message keys across all ratchet steps. The
# specification bounds skips per ratchet step; this bounds the aggregate, so a
# peer cycling DH keys cannot grow the store without limit. Exceeding it is an
# error, never a silent drop: silently discarding a key makes the matching
# message permanently undecryptable.
MAX_STORED_SKIPPED = 4 * MAX_SKIP

PROTOCOL_VERSION = 2
HKDF_INFO = b"WhisperRatchet-v2"

NONCE_SIZE = 24  # XChaCha20-Poly1305 nonce
KEY_SIZE = 32
_DH_PUB_LEN = 32
_ZERO = b'\x00' * KEY_SIZE
_UINT32_MAX = 2 ** 32


class DoubleRatchetError(Exception):
    """Base class for Double Ratchet failures."""


class TooManySkippedMessagesError(DoubleRatchetError):
    """Raised when a peer asks us to skip further than the configured limit."""


class DuplicateMessageError(DoubleRatchetError):
    """Raised when receiving a message that was already processed."""


class InvalidSessionStateError(DoubleRatchetError):
    """Raised when the session state cannot perform the requested operation."""


class DecryptionError(DoubleRatchetError):
    """Raised when AEAD decryption fails."""


class IncompatibleSessionError(DoubleRatchetError):
    """Raised when loading a session written by a different protocol version."""


@dataclass
class SessionState:
    """Complete state of a Double Ratchet session."""

    dh_local_priv: bytes
    dh_local_pub: bytes
    dh_remote_pub: Optional[bytes]

    root_key: bytes
    send_chain_key: Optional[bytes]
    recv_chain_key: Optional[bytes]

    send_msg_count: int = 0
    recv_msg_count: int = 0
    prev_send_count: int = 0

    # (remote DH public key, message number) -> message key. Ordered so the
    # oldest entries are identifiable and eviction is deterministic.
    skipped_keys: "OrderedDict[Tuple[bytes, int], bytes]" = field(
        default_factory=OrderedDict
    )

    def serialize(self, include_message_keys: bool = False) -> dict:
        """
        Serialise to a JSON-friendly dictionary.

        Message keys are omitted unless ``include_message_keys`` is explicitly
        requested. A receiver that reloads without them loses the ability to
        decrypt messages that are still in flight, which is the intended
        trade-off: see the module docstring.
        """
        state = {
            'version': PROTOCOL_VERSION,
            'dh_local_priv': self.dh_local_priv.hex(),
            'dh_local_pub': self.dh_local_pub.hex(),
            'dh_remote_pub': self.dh_remote_pub.hex() if self.dh_remote_pub else None,
            'root_key': self.root_key.hex(),
            'send_chain_key': self.send_chain_key.hex() if self.send_chain_key else None,
            'recv_chain_key': self.recv_chain_key.hex() if self.recv_chain_key else None,
            'send_msg_count': self.send_msg_count,
            'recv_msg_count': self.recv_msg_count,
            'prev_send_count': self.prev_send_count,
        }
        if include_message_keys:
            # A list of records, not a dict: tuple keys are not JSON-safe.
            state['skipped_keys'] = [
                [dh_pub.hex(), msg_num, message_key.hex()]
                for (dh_pub, msg_num), message_key in self.skipped_keys.items()
            ]
        return state

    @classmethod
    def deserialize(cls, data: dict) -> 'SessionState':
        """
        Rebuild state from :meth:`serialize` output.

        Raises:
            IncompatibleSessionError: the payload has no version marker or a
                different one. Earlier versions used a different ``KDF_CK`` and a
                different HKDF info string, so their keys cannot be interpreted
                under this version.
        """
        version = data.get('version')
        if version is None:
            raise IncompatibleSessionError(
                'session state has no protocol version; it predates the v2 KDF '
                'and cannot be loaded safely'
            )
        if version != PROTOCOL_VERSION:
            raise IncompatibleSessionError(
                f'session state version {version} != {PROTOCOL_VERSION}'
            )

        skipped: "OrderedDict[Tuple[bytes, int], bytes]" = OrderedDict()
        for record in data.get('skipped_keys') or []:
            if len(record) != 3:
                raise ValueError('malformed skipped_keys record')
            dh_pub, msg_num, message_key_hex = record
            skipped[(bytes.fromhex(dh_pub), int(msg_num))] = bytes.fromhex(
                message_key_hex
            )

        return cls(
            dh_local_priv=bytes.fromhex(data['dh_local_priv']),
            dh_local_pub=bytes.fromhex(data['dh_local_pub']),
            dh_remote_pub=bytes.fromhex(data['dh_remote_pub']) if data['dh_remote_pub'] else None,
            root_key=bytes.fromhex(data['root_key']),
            send_chain_key=bytes.fromhex(data['send_chain_key']) if data['send_chain_key'] else None,
            recv_chain_key=bytes.fromhex(data['recv_chain_key']) if data['recv_chain_key'] else None,
            send_msg_count=int(data['send_msg_count']),
            recv_msg_count=int(data['recv_msg_count']),
            prev_send_count=int(data['prev_send_count']),
            skipped_keys=skipped,
        )


class DoubleRatchet:
    """
    Double Ratchet with header authentication, skipped-key storage and
    rollback of failed decryptions.
    """

    def __init__(self,
                 dh_private: public.PrivateKey,
                 remote_dh_public: Optional[bytes],
                 root_key: bytes,
                 is_initiator: bool = True):
        if len(root_key) != KEY_SIZE:
            raise InvalidSessionStateError(
                f'root key must be {KEY_SIZE} bytes'
            )
        self._state = SessionState(
            dh_local_priv=bytes(dh_private),
            dh_local_pub=bytes(dh_private.public_key),
            dh_remote_pub=remote_dh_public,
            root_key=root_key,
            send_chain_key=None,
            recv_chain_key=None,
        )
        self._dh_private = dh_private

        # The initiator ratchets first; the responder waits for the first
        # message and learns the initiator's current DH key from its header.
        if is_initiator and remote_dh_public is not None:
            self._perform_dh_ratchet_as_sender()

    @property
    def state(self) -> SessionState:
        return self._state

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _dh(self, private: public.PrivateKey, pub: bytes) -> bytes:
        """X25519 with an explicit degenerate-output check."""
        if len(pub) != _DH_PUB_LEN:
            raise DecryptionError('DH public key must be 32 bytes')
        try:
            shared = nacl.bindings.crypto_scalarmult(bytes(private), pub)
        except Exception as exc:
            raise DecryptionError('X25519 rejected the public key') from exc
        if shared == _ZERO:
            raise DecryptionError(
                'degenerate X25519 output: public key is a low-order point'
            )
        return shared

    def _kdf_rk(self, root_key: bytes, dh_output: bytes) -> Tuple[bytes, bytes]:
        """
        ``KDF_RK``: HKDF-SHA256 with the root key as salt and the DH output as
        IKM, expanded to 64 bytes. The first half is the new root key, the
        second the new chain key.

        The expansion must actually produce 64 bytes. The previous
        implementation took a single 32-byte HMAC and then read ``okm[32:64]``
        out of it, which is always empty, so every chain key was ``b''`` and all
        sessions derived identical message keys.
        """
        okm = hkdf(
            salt=root_key, ikm=dh_output, info=HKDF_INFO + b'\x01', length=2 * KEY_SIZE
        )
        return okm[:KEY_SIZE], okm[KEY_SIZE:2 * KEY_SIZE]

    def _kdf_ck(self, chain_key: bytes) -> Tuple[bytes, bytes]:
        """
        ``KDF_CK``: ``(next_chain_key, message_key)``.

        Per the specification the next chain key is ``HMAC(ck, 0x01)`` and the
        message key is ``HMAC(ck, 0x02)``.
        """
        next_chain_key = hmac.new(chain_key, b'\x01', hashlib.sha256).digest()
        message_key = hmac.new(chain_key, b'\x02', hashlib.sha256).digest()
        return next_chain_key, message_key

    # ------------------------------------------------------------------
    # State snapshot / rollback
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        return {
            'dh_local_priv': self._state.dh_local_priv,
            'dh_local_pub': self._state.dh_local_pub,
            'dh_remote_pub': self._state.dh_remote_pub,
            'root_key': self._state.root_key,
            'send_chain_key': self._state.send_chain_key,
            'recv_chain_key': self._state.recv_chain_key,
            'send_msg_count': self._state.send_msg_count,
            'recv_msg_count': self._state.recv_msg_count,
            'prev_send_count': self._state.prev_send_count,
            'skipped_keys': OrderedDict(self._state.skipped_keys),
        }

    def _restore(self, snapshot: dict) -> None:
        self._state.dh_local_priv = snapshot['dh_local_priv']
        self._state.dh_local_pub = snapshot['dh_local_pub']
        self._state.dh_remote_pub = snapshot['dh_remote_pub']
        self._state.root_key = snapshot['root_key']
        self._state.send_chain_key = snapshot['send_chain_key']
        self._state.recv_chain_key = snapshot['recv_chain_key']
        self._state.send_msg_count = snapshot['send_msg_count']
        self._state.recv_msg_count = snapshot['recv_msg_count']
        self._state.prev_send_count = snapshot['prev_send_count']
        self._state.skipped_keys = snapshot['skipped_keys']
        self._dh_private = public.PrivateKey(self._state.dh_local_priv)

    # ------------------------------------------------------------------
    # Ratchet steps
    # ------------------------------------------------------------------

    def _perform_dh_ratchet_as_sender(self) -> None:
        new_dh_private = public.PrivateKey.generate()
        self._dh_private = new_dh_private
        self._state.dh_local_priv = bytes(new_dh_private)
        self._state.dh_local_pub = bytes(new_dh_private.public_key)

        if self._state.dh_remote_pub is None:
            raise InvalidSessionStateError('remote DH public key required to ratchet')

        dh_output = self._dh(new_dh_private, self._state.dh_remote_pub)
        self._state.root_key, self._state.send_chain_key = self._kdf_rk(
            self._state.root_key, dh_output
        )
        self._state.send_msg_count = 0

    def _perform_dh_ratchet_as_receiver(self, new_remote_dh: bytes) -> None:
        """
        Perform a DH ratchet after a header advertised a new remote key.

        Skipped keys from earlier ratchet steps are intentionally *kept*: they
        are still needed to decrypt messages that were sent before this ratchet
        and arrive afterwards. They are bounded by ``MAX_STORED_SKIPPED``.

        The peer's previous chain is skipped using that chain's own key and
        counter, and ``Nr`` is then reset to 0. Advancing ``Nr`` while skipping
        the old chain would make the first message of the new chain (n = 0)
        look like a replay.
        """
        self._state.prev_send_count = self._state.send_msg_count
        self._state.send_msg_count = 0

        old_chain_key = self._state.recv_chain_key
        old_msg_count = self._state.recv_msg_count
        old_remote_dh = self._state.dh_remote_pub

        if old_chain_key is not None:
            # Messages the peer already sent on the previous chain must become
            # decryptable before the chains move on.
            self._derive_skipped_keys(
                old_chain_key, old_msg_count, self._state.prev_send_count,
                old_remote_dh,
            )

        # The new receiving chain starts at message number 0.
        self._state.recv_msg_count = 0

        dh_output = self._dh(self._dh_private, new_remote_dh)
        self._state.root_key, self._state.recv_chain_key = self._kdf_rk(
            self._state.root_key, dh_output
        )
        self._state.dh_remote_pub = new_remote_dh

        # Second ratchet step: our new sending chain for the peer's new key.
        new_dh_private = public.PrivateKey.generate()
        self._dh_private = new_dh_private
        self._state.dh_local_priv = bytes(new_dh_private)
        self._state.dh_local_pub = bytes(new_dh_private.public_key)

        dh_output_2 = self._dh(new_dh_private, new_remote_dh)
        self._state.root_key, self._state.send_chain_key = self._kdf_rk(
            self._state.root_key, dh_output_2
        )

    def _store_skipped_key(self, key_id: Tuple[bytes, int], message_key: bytes) -> None:
        if len(self._state.skipped_keys) >= MAX_STORED_SKIPPED and \
                key_id not in self._state.skipped_keys:
            raise TooManySkippedMessagesError(
                f'storing skipped keys would exceed MAX_STORED_SKIPPED='
                f'{MAX_STORED_SKIPPED}'
            )
        self._state.skipped_keys[key_id] = message_key

    def _derive_skipped_keys(
        self,
        chain_key: bytes,
        from_msg_num: int,
        to_msg_num: int,
        remote_dh: Optional[bytes],
    ) -> Tuple[bytes, int]:
        """
        Derive and retain the message keys for ``[from_msg_num, to_msg_num)``.

        Takes and returns the chain key and counter explicitly instead of
        touching session state, so a caller can walk an *old* chain without
        disturbing the counter of the current one.

        Raises:
            TooManySkippedMessagesError: the requested gap exceeds ``MAX_SKIP``,
                or the aggregate store is full. Never silently drops a key: a
                dropped key makes the matching message permanently
                undecryptable.
        """
        if to_msg_num <= from_msg_num:
            return chain_key, from_msg_num

        if (to_msg_num - from_msg_num) > MAX_SKIP:
            raise TooManySkippedMessagesError(
                f'skipping {to_msg_num - from_msg_num} messages exceeds '
                f'MAX_SKIP={MAX_SKIP}'
            )
        if len(self._state.skipped_keys) + (to_msg_num - from_msg_num) > MAX_STORED_SKIPPED:
            raise TooManySkippedMessagesError(
                f'storing skipped keys would exceed MAX_STORED_SKIPPED='
                f'{MAX_STORED_SKIPPED}'
            )

        key = chain_key
        msg_num = from_msg_num
        while msg_num < to_msg_num:
            key, message_key = self._kdf_ck(key)
            self._store_skipped_key((bytes(remote_dh) if remote_dh else b'', msg_num),
                                    message_key)
            msg_num += 1
        return key, msg_num

    def _skip_message_keys(self, until: int) -> None:
        """
        Advance the *current* receiving chain to message number ``until``.

        Raises:
            TooManySkippedMessagesError: the peer asked us to skip further than
                ``MAX_SKIP``, or the aggregate store is full.
        """
        if self._state.recv_chain_key is None:
            return
        self._state.recv_chain_key, self._state.recv_msg_count = (
            self._derive_skipped_keys(
                self._state.recv_chain_key,
                self._state.recv_msg_count,
                until,
                self._state.dh_remote_pub,
            )
        )

    # ------------------------------------------------------------------
    # Header handling
    # ------------------------------------------------------------------

    def _validate_header(self, header: dict) -> Tuple[bytes, int]:
        if not isinstance(header, dict):
            raise InvalidSessionStateError('header must be a dict')
        for field_name in ('dh', 'pn', 'n'):
            if field_name not in header:
                raise InvalidSessionStateError(f'header missing {field_name!r}')

        dh_hex = header['dh']
        if not isinstance(dh_hex, str) or len(dh_hex) != _DH_PUB_LEN * 2:
            raise InvalidSessionStateError('header dh must be 32-byte hex')

        for field_name in ('pn', 'n'):
            value = header[field_name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidSessionStateError(
                    f'header {field_name} must be an int'
                )
            if not 0 <= value < _UINT32_MAX:
                raise InvalidSessionStateError(
                    f'header {field_name} out of range'
                )

        return bytes.fromhex(dh_hex), int(header['n'])

    def _serialize_header(self, header: dict) -> bytes:
        """Deterministic serialisation of the header, used as AD."""
        dh = bytes.fromhex(header['dh'])
        return dh + header['pn'].to_bytes(4, 'big') + header['n'].to_bytes(4, 'big')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes, associated_data: bytes = b'') -> Tuple[bytes, dict]:
        """
        Encrypt one message.

        Returns:
            ``(nonce || ciphertext, header)``. The header travels in the clear
            but is bound to the ciphertext as AEAD associated data, so it
            cannot be modified.
        """
        if self._state.send_chain_key is None:
            self._perform_dh_ratchet_as_sender()

        self._state.send_chain_key, message_key = self._kdf_ck(
            self._state.send_chain_key
        )

        header = {
            'dh': self._state.dh_local_pub.hex(),
            'pn': self._state.prev_send_count,
            'n': self._state.send_msg_count,
        }
        ad = associated_data + self._serialize_header(header)

        nonce = nacl.utils.random(NONCE_SIZE)
        ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            plaintext, ad, nonce, message_key
        )

        self._state.send_msg_count += 1
        return nonce + ciphertext, header

    def decrypt(self, encrypted: bytes, header: dict, associated_data: bytes = b'') -> bytes:
        """
        Decrypt one message.

        The header is validated, then the state is snapshotted. Any failure
        restores the snapshot, so a tampered packet, a replay, or a forged header
        carrying an attacker-chosen DH key cannot advance the chain or destroy
        the session.

        Raises:
            InvalidSessionStateError: the header is malformed.
            TooManySkippedMessagesError: the peer exceeded the skip budget.
            DuplicateMessageError: the message was already processed.
            DecryptionError: AEAD verification failed.
        """
        remote_dh, msg_num = self._validate_header(header)
        if len(encrypted) <= NONCE_SIZE:
            raise DecryptionError('ciphertext is too short to contain a nonce')

        ad = associated_data + self._serialize_header(header)
        nonce = encrypted[:NONCE_SIZE]
        ciphertext = encrypted[NONCE_SIZE:]

        snapshot = self._snapshot()
        try:
            return self._decrypt_inner(remote_dh, msg_num, nonce, ciphertext, ad)
        except Exception:
            # Roll back: a message we could not authenticate must leave no
            # trace in the session state.
            self._restore(snapshot)
            raise

    def _decrypt_inner(self, remote_dh: bytes, msg_num: int,
                       nonce: bytes, ciphertext: bytes, ad: bytes) -> bytes:
        # 1. A key retained for an out-of-order message.
        key_id = (remote_dh, msg_num)
        retained = self._state.skipped_keys.get(key_id)
        if retained is not None:
            del self._state.skipped_keys[key_id]
            return self._do_decrypt(retained, nonce, ciphertext, ad)

        # 2. New DH key advertised by the peer.
        current_dh = self._state.dh_remote_pub
        if current_dh is None or not hmac.compare_digest(remote_dh, current_dh):
            self._perform_dh_ratchet_as_receiver(remote_dh)

        # 3. Reject replays before deriving anything.
        if msg_num < self._state.recv_msg_count:
            raise DuplicateMessageError(
                f'message {msg_num} already received '
                f'(current: {self._state.recv_msg_count})'
            )

        # 4. Derive and retain the keys we skipped over.
        self._skip_message_keys(msg_num)

        if self._state.recv_chain_key is None:
            raise InvalidSessionStateError('receive chain key is None')

        # 5. Derive this message's key and decrypt.
        self._state.recv_chain_key, message_key = self._kdf_ck(
            self._state.recv_chain_key
        )
        self._state.recv_msg_count += 1
        return self._do_decrypt(message_key, nonce, ciphertext, ad)

    def _do_decrypt(self, message_key: bytes, nonce: bytes,
                    ciphertext: bytes, ad: bytes) -> bytes:
        try:
            return nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
                ciphertext, ad, nonce, message_key
            )
        except Exception as exc:
            raise DecryptionError(
                'AEAD decryption failed: message corrupted or tampered with'
            ) from exc
