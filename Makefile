test:
	pytest -m "not slow"

test-debug:
	pytest -m "not slow" --pdb

test-all:
	pytest