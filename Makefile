.PHONY: test toy calibrate calibrate-v1 ceiling ceiling-phase bundle

test:
	PYTHONPATH=src pytest -q

calibrate:
	python scripts/calibrate_stream.py --seeds 900 901 902

calibrate-v1:
	python scripts/calibrate_interference.py --seeds 900 901 902

toy:
	python scripts/run_toy.py --seeds 0 1 2 3 4

ceiling:
	python scripts/run_ceiling.py --seeds 0 1 2 3 4 5 6 7

ceiling-phase:
	python scripts/run_ceiling_phase.py

bundle:
	git archive --format=zip --output=../modular-consolidation-repo.zip HEAD
