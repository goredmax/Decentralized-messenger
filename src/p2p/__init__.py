"""
Transport layer: the seam between a session and whatever carries its bytes.

What is here now: a :class:`~src.p2p.relay.Transport` interface and an
in-process relay that implements it, so a conversation can be run end to end
and its frame sizes measured.

What is not here, and is claimed nowhere else either: peer discovery, a DHT,
Kademlia, libp2p, NAT traversal, onion services. The previous version of this
file described all of them as what this module "provides", which was false --
it was an empty file. Naming a capability that does not exist is how
``requirements.in`` ended up carrying libp2p, stem, aiohttp and
``sqlcipher-legacy`` for months, none of which anything imported.
"""

__all__ = ['relay']

__version__ = "0.1.0"
