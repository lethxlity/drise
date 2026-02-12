#* Variables
SHELL := /usr/bin/env bash
PYTHON ?= python3
ENV_FILE := .env



#* Initialization
project-init: install-packages tools-install

install-packages:
	poetry install -n

lock-update:
	poetry lock --no-update

requirements-export: lock-update
	poetry export --without-hashes > requirements.txt

requirements-export-dev: lock-update
	poetry export --with dev --without-hashes > requirements.dev.txt

#* Tools
tools-install:
	poetry run pre-commit install --hook-type prepare-commit-msg --hook-type pre-commit


pre-commit-update:
	poetry run pre-commit autoupdate

pre-commit-run-all:
	poetry run pre-commit run --all-files



#* Tests
tests:
	poetry run pytest tests/ -c pyproject.toml

#* Linting
type-check:
	poetry run mypy

#* Cleaning
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E "(.ipynb_checkpoints$$)" | xargs rm -rf

build-remove:
	rm -rf build/

clean-all: pycache-remove build-remove

#* Service targets
grep-todos:
	git grep -EIn "TODO|FIXME|XXX"
