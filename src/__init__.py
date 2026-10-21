"""
The installed application package.

The import name stays ``src`` -- the same one all 460 tests already use
(``from src.crypto.x3dh import ...``). That is deliberate: renaming the package
would touch 76 import lines across 15 test files plus pytest.ini and CI, for a
cosmetic gain nobody sees. The distribution name is a normal one,
``decentralized-messenger``; a package and a distribution are not required to
share a name.
"""
