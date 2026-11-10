.PHONY: install train test lint serve docker

install:
	pip install -r requirements-dev.txt
	pip install -e .

train:
	python -m bcd.train

test:
	pytest

lint:
	ruff check .
	ruff format --check .

serve:
	uvicorn bcd.api:app --reload

docker:
	docker build -t breast-cancer-diagnosis .
	docker run --rm -p 8000:8000 breast-cancer-diagnosis
