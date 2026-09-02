.PHONY: run test eval lint docker
run:
	uvicorn sentinelrag.main:app --reload
test:
	pytest -q
eval:
	python scripts/run_evaluation.py --dataset data/evaluation/demo_cases.jsonl
lint:
	ruff check src tests scripts
docker:
	docker compose up --build

