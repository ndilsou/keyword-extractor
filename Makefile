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

run: port ?= 80
run:
	streamlit run app.py --server.port $(port)

docker-run:
	docker run -it keyword_extractor

build:
	docker build . -t keyword_extractor
