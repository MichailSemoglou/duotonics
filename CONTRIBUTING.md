# Contributing to Duotonics

## Development setup

Duotonics supports Python 3.9 through 3.13. Create a virtual environment, then install the package and development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`.
python -m pip install -e . -r requirements-dev.txt
```

On Python 3.10 or higher, also install the static-analysis and security tools:

```bash
python -m pip install -r requirements-quality.txt
```

## Making changes

1. Create a feature branch from an up-to-date `main`.
2. Keep each change focused and preserve seeded `srgb` output unless a change explicitly documents new compatibility behavior.
3. Add tests for normal, boundary, invalid, and failure cases affected by the change.
4. Update the README, relevant docstrings, and changelog when public behavior changes.
5. Use concise Conventional Commit subjects, such as `feat:`, `fix:`, `docs:`, `test:`, `build:`, or `ci:`.

## Verification

Run the checks that cover your change before opening a pull request:

```bash
pytest --cov --cov-report=term-missing
ruff check .
mypy duotone_generator.py
bandit -r duotone_generator.py
pip-audit
python -m build
python -m twine check dist/*
```

For image-output changes, also run the CLI on a real image and inspect the PNG and CSV sidecars.

## Pull requests

Open a focused pull request against `main`. Describe the problem, acceptance criterion, learning objective, hypothesis, and evidence. Link relevant issues and record the verification commands and results. Keep generated variations, virtual environments, caches, and local configuration out of version control.
