"""
Protocol layer: wire encoding, key verification and group messaging.

Split out from ``src.crypto`` deliberately. The crypto modules implement
primitives and the session state machine; this package is about moving those
structures between processes, which is a different set of concerns and a
different set of mistakes.
"""

__all__ = ['wire']
