"""Paths for results, figures, etc..."""

from pathlib import Path


# where the pyproject.toml lies
_PACKAGE_HOME = Path(__file__).parent.parent.parent

RESULTS_PATH = _PACKAGE_HOME / 'results'  # results of the simulations
