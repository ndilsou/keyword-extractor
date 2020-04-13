lab:
	jupyter lab --notebook-dir notebooks

test: options?=
test: ## Run the test suite
	poetry run python -m pytest $(options) tests/

test-watch: options?=
test-watch: ## Run the test suite
	ptw $(options)
