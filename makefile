up:
	docker compose up -d

down:
	docker compose down

setup:
	docker compose run --rm setup

eval:
	docker compose run --rm eval

dataset ?= scifact

download:
	uv run python -m src.download_data $(dataset)
