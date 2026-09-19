.DEFAULT_GOAL := help
.PHONY: help install install-ml dev test lint format typecheck check seed ingest eval up down logs docker-build pdf clean

PORT ?= 8000
API_URL ?= http://127.0.0.1:$(PORT)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install core + dev dependencies (fast, no ML models)
	uv sync

install-ml: ## Also install sentence-transformers + cross-encoder (torch)
	uv sync --extra ml

dev: ## Run the API + web UI with auto-reload on http://localhost:$(PORT)
	uv run uvicorn hybrid_rag.api.app:app_factory --factory --reload --port $(PORT)

test: ## Run the test suite (offline)
	uv run pytest

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Static type checking
	uv run mypy

check: lint typecheck test ## Everything CI runs

seed: ## Upload sample_docs/ to the running server (make dev / make up first)
	uv run hybrid-rag seed sample_docs --url $(API_URL)

ingest: ## Ingest sample_docs/ directly into ./data (no server needed)
	uv run hybrid-rag ingest sample_docs

eval: ## Retrieval eval: dense vs BM25 vs hybrid vs hybrid+rerank
	uv run hybrid-rag eval

up: ## Start API + Qdrant with docker compose
	docker compose up --build -d
	@echo "UI: $(API_URL)"

down: ## Stop docker compose services
	docker compose down

logs: ## Tail API logs
	docker compose logs -f api

docker-build: ## Build the Docker image
	docker build -t hybrid-rag:latest .

pdf: ## Regenerate the sample PDF
	uv run scripts/make_sample_pdf.py

clean: ## Remove local data and caches
	rm -rf data .pytest_cache .ruff_cache .mypy_cache
