.PHONY: pypi prepare

pypi:
	uv run python scripts/prep_pypi.py

prepare: 
	uv run bash scripts/build.sh