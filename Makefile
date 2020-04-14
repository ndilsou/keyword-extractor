appname = keyword_extractor
hub-imagename = ndilsou/streamlit-keyword-extractor
envs := prod stag test
branch=$$( git rev-parse --abbrev-ref HEAD)
version = $$(cat $(CURDIR)/$(appname)/about.py | grep __version__ | cut -d\' -f2)

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

run-app:
	docker run -it -p 80:80 --name $(appname) $(hub-imagename):latest

build:
	docker build . -t $(appname)

tag: build
	docker tag keyword_extractor $(appname):$(version)
	docker tag keyword_extractor $(hub-imagename):$(version)
	docker tag keyword_extractor $(hub-imagename):latest

push: tag
	docker push $(hub-imagename):$(version)
	docker push $(hub-imagename):latest

