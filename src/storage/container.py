"""
Encrypted, authenticated storage for secrets and session state.

Session state is a decryption capability. ``SessionState`` contains the root key,
the chain keys and the local DH private key, and ``deserialize`` trusts whatever
it is handed. So anyone who can write the state file can roll back the message
counters, replay an old ratchet, or swap a DH key, and the implementation has no
way to notice.

This module is the storage boundary. Every file is sealed with
XChaCha20-Poly1305 under a key derived from a password with scrypt, and the
header is fed in as associated data, so the KDF parameters are authenticated
too. That last part matters: if the cost parameters were outside the AEAD, an
attacker could rewrite the file with ``N = 2**8`` and then brute-force the
password at leisure.

File layout::

    header  = magic(8) | version(1) | kdf_id(1) | N_log2(1) | r(1) | p(1) | salt(16)
    body    = nonce(24) | ciphertext
    payload = epoch(8) | plaintext

The epoch lives *inside* the sealed payload, not in the header. An epoch in the
header would be unauthenticated and could simply be raised by an attacker.

Scope of the rollback protection, stated plainly: the epoch makes tampering
*within* a file detectable, and lets a caller refuse an older file. Detecting
replacement of the whole file with an older but internally valid one needs an
anchor outside this file, such as an OS keystore or a remote counter. There is
no such anchor here, so whole-file rollback by an attacker who can write the
directory is **not** detected. :meth:`Container.load` takes ``minimum_epoch``
precisely so a caller with an external anchor can supply one.
"""

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import nacl.bindings
import nacl.utils

from ..crypto.kdf import hkdf as hkdf_sha256

MAGIC = b'ANARCHY\x01'
CONTAINER_VERSION = 1
KDF_SCRYPT = 1

_HEADER_LEN = len(MAGIC) + 1 + 1 + 1 + 1 + 1 + 16
_NONCE_LEN = 24
_EPOCH_LEN = 8
_TAG_LEN = 16

#: Context string for the HKDF that turns the scrypt output into an AEAD key.
_KEY_INFO = b"anarchy-container-v1"

#: Serialise the epoch big-endian so byte order is not a platform detail.
_EPOCH_STRUCT = struct.Struct('>Q')


class StorageError(Exception):
    """Base class for storage failures."""


class WrongPassword(StorageError):
    """The file did not decrypt under the supplied password."""


class TamperedState(StorageError):
    """The file failed authentication: wrong password, or modified bytes."""


class EpochRollback(StorageError):
    """The file holds an epoch older than the caller's minimum."""


class MalformedContainer(StorageError):
    """The file is not a container, or is truncated beyond use."""


@dataclass(frozen=True)
class KdfParams:
    """scrypt work factors. Stored in the header and authenticated as AD."""

    n_log2: int = 15
    r: int = 8
    p: int = 1

    #: Refuse parameters so weak that the password becomes free to guess.
    MIN_N_LOG2 = 12
    MAX_N_LOG2 = 20

    def __post_init__(self) -> None:
        if not self.MIN_N_LOG2 <= self.n_log2 <= self.MAX_N_LOG2:
            raise ValueError(
                f'n_log2 must be {self.MIN_N_LOG2}..{self.MAX_N_LOG2}, '
                f'got {self.n_log2}'
            )
        if not 1 <= self.r <= 32:
            raise ValueError(f'r out of range: {self.r}')
        if not 1 <= self.p <= 16:
            raise ValueError(f'p out of range: {self.p}')

    @property
    def n(self) -> int:
        return 1 << self.n_log2

    def memory_required(self) -> int:
        """
        Bytes scrypt needs for these parameters, plus headroom.

        scrypt uses roughly ``128 * r * N``. Asking OpenSSL for exactly that
        trips its own limit check, so the figure is rounded up generously.
        """
        return 128 * self.r * self.n + (1 << 22)

    def header_bytes(self) -> bytes:
        return bytes((KDF_SCRYPT, self.n_log2, self.r, self.p))

    @classmethod
    def from_header(cls, raw: bytes) -> 'KdfParams':
        """
        Rebuild parameters read from an untrusted header.

        Out-of-range values become :class:`MalformedContainer` rather than a
        bare ``ValueError``: a header read off disk is not a caller error, and
        the boundary should only speak in storage-domain exceptions.
        """
        if len(raw) != 4:
            raise MalformedContainer('truncated kdf parameters')
        kdf_id, n_log2, r, p = raw
        if kdf_id != KDF_SCRYPT:
            raise MalformedContainer(f'unknown kdf id {kdf_id}')
        try:
            return cls(n_log2=n_log2, r=r, p=p)
        except ValueError as exc:
            raise MalformedContainer(f'implausible kdf parameters: {exc}') from exc


DEFAULT_KDF = KdfParams()


def _derive_key(password: str, salt: bytes, params: KdfParams) -> bytes:
    """
    Derive the AEAD key from a password.

    The scrypt output is passed through HKDF for domain separation, so a key
    derived here can never collide with one derived for another purpose from
    the same password and salt.
    """
    if not isinstance(password, str) or not password:
        raise ValueError('password must be a non-empty string')
    material = hashlib.scrypt(
        password.encode('utf-8'),
        salt=salt,
        n=params.n,
        r=params.r,
        p=params.p,
        dklen=32,
        # OpenSSL applies its own memory ceiling, and the default is low enough
        # to reject the default parameters here. Ask for what the parameters
        # actually need rather than inheriting an unrelated global limit.
        maxmem=params.memory_required(),
    )
    # libsodium's convention: an all-zero key means the input was degenerate.
    if material == b'\x00' * 32:
        raise StorageError('derived key is all zero')
    return hkdf_sha256(
        salt=b'\x00' * 32, ikm=material, info=_KEY_INFO, length=32
    )


class Container:
    """
    One encrypted file holding a versioned payload.

    Usage::

        container = Container.create(path, password)
        container.store(b'payload', epoch=1)
        epoch, payload = container.load()
    """

    def __init__(self, path: str, key: bytes, params: KdfParams, salt: bytes):
        self._path = path
        self._key = key
        self._params = params
        self._salt = salt
        self._header = self._build_header(params, salt)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(params: KdfParams, salt: bytes) -> bytes:
        return MAGIC + bytes((CONTAINER_VERSION,)) + params.header_bytes() + salt

    @classmethod
    def create(
        cls,
        path: str,
        password: str,
        *,
        kdf_params: Optional[KdfParams] = None,
        overwrite: bool = False,
    ) -> 'Container':
        """
        Create a new container file with a fresh random salt.

        Args:
            overwrite: destroy an existing file. This also **resets the epoch to
                0**, because the old state is gone; a caller keeping an
                external epoch anchor must re-anchor afterwards.

        Raises:
            FileExistsError: the file exists and ``overwrite`` is False.
        """
        params = kdf_params or DEFAULT_KDF
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f'{path} already exists')
        salt = nacl.utils.random(16)
        key = _derive_key(password, salt, params)
        container = cls(path, key, params, salt)
        container.store(b'', epoch=0)
        return container

    @classmethod
    def open(cls, path: str, password: str) -> 'Container':
        """
        Open an existing container.

        The header is read and the key derived *before* anything is decrypted,
        so a wrong password fails cheaply rather than after allocating.

        Raises:
            MalformedContainer: the file is not a container.
        """
        raw = cls._read_file(path)
        header, _ = cls._split(raw, path)
        version = header[len(MAGIC)]
        if version != CONTAINER_VERSION:
            raise MalformedContainer(
                f'container version {version} != {CONTAINER_VERSION}'
            )
        params = KdfParams.from_header(header[len(MAGIC) + 1:len(MAGIC) + 5])
        salt = header[-16:]
        key = _derive_key(password, salt, params)
        return cls(path, key, params, salt)

    @classmethod
    def exists(cls, path: str) -> bool:
        return os.path.exists(path)

    # ------------------------------------------------------------------
    # io
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: str) -> bytes:
        if not os.path.exists(path):
            raise MalformedContainer(f'{path} does not exist')
        try:
            with open(path, 'rb') as handle:
                return handle.read()
        except OSError as exc:
            raise StorageError(f'cannot read {path}') from exc

    @staticmethod
    def _split(raw: bytes, path: str) -> Tuple[bytes, bytes]:
        if len(raw) < _HEADER_LEN:
            raise MalformedContainer(
                f'{path} is {len(raw)} bytes, shorter than a header'
            )
        if not raw.startswith(MAGIC):
            raise MalformedContainer(f'{path} is not an anarchy container')
        if len(raw) < _HEADER_LEN + _NONCE_LEN + _TAG_LEN:
            raise MalformedContainer(f'{path} is truncated')
        return raw[:_HEADER_LEN], raw[_HEADER_LEN:]

    def store(self, payload: bytes, epoch: int) -> None:
        """
        Seal ``payload`` under ``epoch`` and write it atomically.

        The epoch must not go backwards for this container; use
        :meth:`load` to read the current value first.

        Raises:
            StorageError: the payload is too large or the write fails.
        """
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError('payload must be bytes')
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError('epoch must be a non-negative int')
        if epoch >= 2 ** 64:
            raise ValueError('epoch does not fit in 64 bits')

        plaintext = _EPOCH_STRUCT.pack(epoch) + bytes(payload)
        nonce = nacl.utils.random(_NONCE_LEN)
        ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
            plaintext, self._header, nonce, self._key
        )
        blob = self._header + nonce + ciphertext

        directory = os.path.dirname(os.path.abspath(self._path)) or '.'
        os.makedirs(directory, exist_ok=True)
        # Write beside the target then rename, so a crash mid-write leaves the
        # previous good file rather than a half-written one.
        temporary = os.path.join(directory, f'.{os.path.basename(self._path)}.tmp')
        try:
            with open(temporary, 'wb') as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        except OSError as exc:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
            raise StorageError(f'cannot write {self._path}') from exc

    def load(self, minimum_epoch: Optional[int] = None) -> Tuple[int, bytes]:
        """
        Open the container and return ``(epoch, payload)``.

        Raises:
            WrongPassword: authentication failed.
            MalformedContainer: the file is truncated or not a container.
            EpochRollback: the stored epoch is below ``minimum_epoch``.
        """
        raw = self._read_file(self._path)
        header, body = self._split(raw, self._path)

        nonce = body[:_NONCE_LEN]
        ciphertext = body[_NONCE_LEN:]
        try:
            plaintext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
                ciphertext, header, nonce, self._key
            )
        except Exception as exc:
            # AEAD cannot distinguish a wrong password from a modified file,
            # and must not try to.
            raise WrongPassword(
                f'cannot decrypt {self._path}: wrong password or modified file'
            ) from exc

        if len(plaintext) < _EPOCH_LEN:
            raise MalformedContainer('decrypted payload is too short to hold an epoch')
        (epoch,) = _EPOCH_STRUCT.unpack(plaintext[:_EPOCH_LEN])
        payload = plaintext[_EPOCH_LEN:]

        if minimum_epoch is not None and epoch < minimum_epoch:
            raise EpochRollback(
                f'stored epoch {epoch} is older than the required {minimum_epoch}'
            )
        return epoch, payload

    def delete(self) -> None:
        """Remove the file. Missing file is not an error."""
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise StorageError(f'cannot delete {self._path}') from exc
