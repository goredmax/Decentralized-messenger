"""The runtime dependency list lives in two files, and both are load-bearing.

``pyproject.toml`` is what pip reads when the package is installed;
``requirements.in`` is what pip-compile turns into the lock CI installs. Both
files carry a comment saying the lists must stay identical, but a comment is
not a check: adding a dependency to one and forgetting the other produces two
install paths that quietly disagree. The package could work in CI (which uses
the lock) and fail for someone who ran ``pip install .``, or the other way
round.

This test turns the comment into an executable assertion. It deliberately
compares the declared requirements, not resolved versions: version selection
belongs to the lock files, which pip-compile resolves on the minimum supported
interpreter.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def requirements_in() -> list[str]:
    """Return the runtime requirements as pip-compile sees them.

    ``requirements.in`` is a plain requirements file: blank lines and comments
    are ignored. Lines starting with ``-`` are options (for example ``-r``) and
    are not dependencies themselves.
    """
    requirements = []
    for raw in (ROOT / "requirements.in").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            requirements.append(line)
    return requirements


def test_pyproject_declares_the_same_runtime_dependencies() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["project"]["dependencies"]

    assert declared == requirements_in(), (
        "pyproject.toml [project].dependencies and requirements.in disagree; "
        "pip installs the first and pip-compile locks the second"
    )
