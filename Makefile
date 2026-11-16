.PHONY: test verify reproduce paper package

test:
	python -m pytest tests -q

verify:
	python -m names_mechinterp.verify --verify-compressed

reproduce:
	python -m names_mechinterp.reproduce --data-root data/processed --output results/reproduced.json

paper:
	python paper/build_submission.py

package:
	python tools/package_submission.py
