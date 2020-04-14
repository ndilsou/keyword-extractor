setup:
	pipenv install --dev
	spacy download en_core_web_sm

lab:
	jupyter lab --notebook-dir notebooks

test: options?=
test: ## Run the test suite
	pipenv run python -m pytest $(options) tests/

test-watch: options?=
test-watch: ## Run the test suite
	ptw $(options)

run:
	streamlit run app.py

build:
	docker build . -t keyword_extractor
