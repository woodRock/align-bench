# Local equivalents of the CI jobs. `make check` runs all three.
.PHONY: venv install check lint types test test-fast

# alignbench lives in its own venv: the global python3.11 has editable warpalign, jointalign and
# trustalign checkouts, and the [golden] extra would replace them with pinned commits.
PY := .venv/bin/python

venv:
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

install:
	$(PY) -m pip install -e ".[dev]"

check: lint types test

lint:
	$(PY) -m ruff check .

types:
	$(PY) -m mypy

test:
	$(PY) -m coverage run -m pytest && $(PY) -m coverage report

test-fast:
	$(PY) -m pytest -m "not slow"
