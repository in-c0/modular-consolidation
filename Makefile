.PHONY: test toy calibrate bundle

test:
	PYTHONPATH=src pytest -q

calibrate:
	python scripts/calibrate_stream.py --seeds 900 901 902

toy:
	python scripts/run_toy.py --seeds 0 1 2 3 4

bundle:
	git archive --format=zip --output=../modular-consolidation-repo.zip HEAD
