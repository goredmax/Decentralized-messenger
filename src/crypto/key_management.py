"""
Key management helpers for E2EE.

The X3DH identity is an Ed25519 key pair, and the protocol also needs the
matching X25519 keys for Diffie-Hellman. This module holds only that conversion:
keeping it in one place makes the deviation from Signal -- one key used for both
signing and DH -- visible instead of scattering conversions through the codebase.
"""

import nacl.signing as signing
from nacl.public import PrivateKey, PublicKey


def identity_private_x25519(signing_key: signing.SigningKey) -> PrivateKey:
    """
    Derive the X25519 private key matching an Ed25519 identity key.

    Deviation from the Signal specification, which keeps separate keys for
    signing (Ed25519) and for DH (X25519). Reusing one key for both roles means
    a single key compromise breaks both authentication and key agreement, and
    the Ed25519 -> X25519 conversion is only defined for curve points. This is
    centralised here so the deviation stays visible and has one implementation.
    """
    return PrivateKey(signing_key.to_curve25519_private_key().encode())


def identity_public_x25519(verify_key: signing.VerifyKey) -> PublicKey:
    """Derive the X25519 public key matching an Ed25519 identity key."""
    return PublicKey(verify_key.to_curve25519_public_key().encode())
