from typing import Dict, Sequence, Iterable
from zipfile import ZipFile

from preshed.maps import PreshMap
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
import streamlit as st

from keyword_extractor import extractors
from keyword_extractor.corpus import Corpus, create_spacy_corpus, create_text_corpus_from_zipfile
from keyword_extractor.extractors import KeywordExtractor, ExtractionResult, DocMatch, SentenceMatch
from keyword_extractor.utils import summarize_text, highlight_sentence

HTML_SPACE = '&nbsp;' # To help with markdown formating
HASH_FUNCS = {Language: id, Corpus: hash}


def main():
    with st.spinner(text='Loading NLP model...'):
        nlp = get_spacy_lang()
    st.title('Keyword Extractor')
    store = get_state_store()
    store['nlp'] = nlp
    sidebar(store)

    corpus = store.get('corpus')
    if corpus:
        document_display(store['fileids'], corpus)
        result = extract_keywords(corpus, store['model_params'])
        extraction_report(result)


def document_display(fileids: Sequence[str], corpus: Corpus):
    selection = st.multiselect("Select which document to show", fileids)

    for doc in corpus.get(lambda doc: doc._.meta['fileid'] in selection):
        fileid = doc._.meta['fileid']
        st.subheader(fileid)

        if st.checkbox('Full text', key=f'summary-toggle-{fileid}'):
            st.text(doc)
        else:
            st.text(summarize_text(doc))


def sidebar(store: dict):
    '''
    A component for all the elements of the sidebar.
    '''

    file_uploader(store)
    model_selector(store)


def file_uploader(store: dict):
    '''
    Component handling file upload and creation of the text corpus.
    '''

    result = st.sidebar.file_uploader("Upload Corpus", type="zip")

    if result:
        if not 'text_corpus' in store:
            text_corpus = create_text_corpus_from_zipfile(ZipFile(result))
            store['text_corpus'] = text_corpus
            store['fileids'] = text_corpus.fileids()
            nlp = store['nlp']
            corpus = create_spacy_corpus(text_corpus, nlp)
            store['corpus'] = corpus
    else:
        st.sidebar.info("Upload one or more `.zip` files.")

    if st.sidebar.button("Clear file"):
        store.pop('text_corpus', None)
        store.pop('corpus', None)

    corpus = store.get('text_corpus')
    if corpus:

        if st.sidebar.checkbox("List documents in corpus?", True):
            st.sidebar.dataframe(pd.Series(corpus.fileids(), name='documents'))


def model_selector(store: dict):
    '''
    Allows user to select a model and possibly some hyperparameters.
    '''
    model = st.sidebar.selectbox('Model', ['TfIdf', 'TextRank', 'EmbedRank'])
    topn = st.sidebar.number_input('TopN to extract per document:', value=5, min_value=1)
    store['model_params'] = {'model': model, 'topn': topn}


@st.cache(hash_funcs={Language: id, Corpus: hash}, allow_output_mutation=True)
def extract_keywords(corpus: Corpus, params: dict) -> ExtractionResult:
    klass = {
        'TfIdf': extractors.TfIdfKeywordExtractor,
        'TextRank': extractors.TextRankKeywordExtractor,
        'EmbedRank': extractors.EmbedRankKeywordExtractor
    }.get(params['model'], None)
    extractor = klass(params['topn'])
    with st.spinner(text='Training in progress...'):
        extractor = fit_model(extractor, corpus)

    with st.spinner(text='Extraction in progress...'):
        result = extractor.extract(corpus)

    return result


@st.cache(hash_funcs=HASH_FUNCS, allow_output_mutation=True)
def fit_model(model: KeywordExtractor, corpus: Corpus) -> KeywordExtractor:
    return model.fit(corpus)


def extraction_report(result: ExtractionResult):
    st.header('Extraction results')
    choices = {
        'Summary table': summary_table,
        'Full report': full_report
    }
    choice = st.selectbox('Output', list(choices.keys()))
    output = choices[choice]
    output(result)


def summary_table(result: ExtractionResult):
    st.subheader('Keywords with scores and parent documents')
    scores = result.scores
    aggregate_scores = {
        kw: {
            'average_score': np.mean(list(v.values())),
            'documents': ', '.join(v.keys())
        }
        for kw, v
        in scores.items()
    }
    df = pd.DataFrame \
        .from_dict(aggregate_scores, orient='index') \
        .sort_values('average_score', ascending=False)
    st.table(df)


def full_report(result: ExtractionResult):
    '''
    Show the report for each keywords with optional filtering.
    '''
    keywords = list(result.raw.keys())
    selected_keywords = st.multiselect("Filter keywords", keywords)

    for kw, matches in result.raw.items():
        if selected_keywords and kw not in selected_keywords:
            continue

        keyword_report(kw, matches)


def keyword_report(keyword, doc_matches: Iterable[DocMatch]):
    '''
    Show the report for a given keyword.
    '''

    st.markdown('____')
    st.markdown(f'#### {keyword}')

    for doc_match in doc_matches:
        st.markdown(f'- ##### file: {doc_match.fileid}')
        st.markdown(f'- ##### score: {doc_match.score:.3f}')
        for sent_match in doc_match.sents:
            highlighted_sent = highlight_sentence(sent_match.text, sent_match.start_char, sent_match.end_char)
            st.markdown(f'> {highlighted_sent}')


@st.cache(hash_funcs=HASH_FUNCS, allow_output_mutation=True)
def get_spacy_lang() -> Language:
    return spacy.load('en_core_web_sm')


@st.cache(allow_output_mutation=True)
def get_state_store() -> dict:
    """This dictionary is initialized once and can be used to store the state of the application in memory"""

    return {}


if __name__ == "__main__":
    main()
