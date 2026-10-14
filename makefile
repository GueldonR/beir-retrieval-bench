up:
	docker compose up -d

down:
	docker compose down

setup:
	uv run python -m src.setup_infra

eval:
	uv run python -m src.beir_test
