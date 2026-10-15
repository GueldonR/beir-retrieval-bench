up:
	docker compose up -d

down:
	docker compose down

setup:
	uv run python -m src.setup_infra

eval:
	docker compose run --rm eval

dataset ?= scifact

download:
	uv run python -m src.download_data $(dataset)
