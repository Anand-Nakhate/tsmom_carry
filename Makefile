.PHONY: install download process clean test help

PYTHON := python3
PIP := pip3

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt

download: ## Run the bulk downloader
	$(PYTHON) src/download.py

process: ## Run the data processor
	$(PYTHON) src/process_dataset.py

test: ## Run tests
	$(PYTHON) -m pytest tests/

clean: ## Remove artifacts and cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf logs/
