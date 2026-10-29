PY := .venv/bin/python

.PHONY: test report report-check smoke fixture

test:
	$(PY) -m pytest tests/ -q

report:
	PYTHONPATH=. $(PY) analysis/report.py --results results/fixture --out analysis/out
	PYTHONPATH=. $(PY) analysis/report.py --results results/qwen32-1b-cpu --out analysis/out

report-check:
	PYTHONPATH=. $(PY) analysis/report.py --results results/fixture --out analysis/out --check

fixture:
	PYTHONPATH=. $(PY) analysis/make_fixture.py

smoke:
	$(PY) analysis/smoke.py
