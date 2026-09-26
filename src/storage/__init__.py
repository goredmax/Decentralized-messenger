"""
Encrypted, persistent storage for keys and session state.

``src.crypto`` keeps everything in memory, which is fine for tests and useless
in practice: the prekey pool dies with the process, so after a restart the only
reachable handshake is the 3-DH variant, which has no forward secrecy.

:mod:`~src.storage.container` provides the sealed file format,
:mod:`~src.storage.store` the stores built on it. Read both module docstrings
for what is and is not protected: the stores are single-process atomic, and
whole-file rollback by an attacker with write access is not detected.
"""

__all__ = ['container', 'store']
