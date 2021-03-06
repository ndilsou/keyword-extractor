FROM python:3.7

ENV STREAMLIT_PORT=80
ENV STREAMLIT_SERVER_PORT=80

RUN apt-get update -y \
      && apt-get install -y build-essential openssl \
      && pip install -U pip \
      && pip install pipenv

WORKDIR /usr/app/
COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy \
        && spacy download en_core_web_sm

COPY keyword_extractor ./keyword_extractor
COPY app.py Makefile ./

EXPOSE $PORT
CMD ["make", "run"]
