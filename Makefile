.PHONY: install test train predict backtest clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

train:
	python -m src.cli train

predict:
	python -m src.cli predict

backtest:
	python -m src.cli backtest

clean:
	rm -rf models/ data_cache/ plots/ __pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
