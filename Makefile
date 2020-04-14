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

run: PORT ?= 80
run:
	streamlit run app.py --server.port $(PORT)

run-app:
	docker run -it -p 9888:8888 -e PORT=9888 --name $(appname) $(appname):latest

build:
	docker build . -t $(appname)

tag: build
	docker tag $(appname) $(appname):$(version)
	docker tag $(appname) $(hub-imagename):$(version)
	docker tag $(appname) $(hub-imagename):latest


push: tag
	docker push $(hub-imagename):$(version)
	docker push $(hub-imagename):latest

heroku-deploy: build
	docker tag $(appname) registry.heroku.com/$(app)/web
	heroku container:login
	heroku container:push web -a $(app)
	heroku container:release web -a $(app)



