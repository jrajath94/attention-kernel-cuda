.PHONY: install test bench run lint clean

install:
	pip install -e ".[dev,bench]"

test:
	pytest tests/ -v --tb=short --cov=src/attention_kernel_cuda --cov-report=term-missing --cov-report=xml

bench:
	python benchmarks/bench_core.py

run:
	python examples/quickstart.py

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
