"""
Tor integration module.

An empty stub. The previous version of this file claimed to provide anonymous
communication through Tor hidden services; it provided nothing and imported
nothing. Tor is not implemented, not tested, and not planned in this phase --
see docs/SECURITY_BOUNDARIES.md section 5 for what depends on it and why the
order is transport first, metadata second.

Nothing imports this module. It is kept because the package lists it, and
dropping it would be a statement about the roadmap rather than a fix.
"""

__version__ = "0.1.0"
