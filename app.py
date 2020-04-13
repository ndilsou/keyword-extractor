from typing import Dict, Sequence
from zipfile import ZipFile

from preshed.maps import PreshMap
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
import streamlit as st

from keyword_extractor import extractors
from keyword_extractor.corpus import Corpus, create_spacy_corpus, create_text_corpus_from_zipfile
from keyword_extractor.extractors import KeywordExtractor
from keyword_extractor.utils import summarize_text

HTML_SPACE = '&nbsp;' # To help with markdown formating

def main():
    with st.spinner(text='Loading NLP model...'):
        nlp = get_spacy_lang()
    st.title('Keyword Extractor')
    store = get_state_store()
    store['nlp'] = nlp
    sidebar(store)

    if (corpus:= store.get('corpus')):
        document_display(store['fileids'], corpus)
        extractor = fit_model(corpus, store['selected_model'])
        extraction_results(extractor)
        # keyword_table(extractor)


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

    if (selected_model := st.sidebar.selectbox('Model', ['TfIdf', 'TextRank', 'EmbedRank'])):
        store['selected_model'] = selected_model


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

    if (corpus := store.get('text_corpus')):

        if st.sidebar.checkbox("List documents in corpus?", True):
            st.sidebar.dataframe(pd.Series(corpus.fileids(), name='documents'))


def fit_model(corpus: Corpus, selected_model: str, topn: int = 5) -> KeywordExtractor:
    klass = {
        'TfIdf': extractors.TfIdfKeywordExtractor
    }.get(selected_model, None)
    extractor = klass(topn)
    with st.spinner(text='Training in progress...'):
        extractor.fit(corpus)

    return extractor


def extraction_results(extractor: KeywordExtractor):
    st.header('Extraction results')
    choices = {
        'Summary table': summary_table,
        'Full report': full_report
    }
    choice = st.selectbox('Output', list(choices.keys()))
    output = choices[choice]
    output(extractor)


def summary_table(extractor: KeywordExtractor):
    st.subheader('Keywords with scores and parent documents')
    scores = extractor.scores
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


def full_report(extractor: KeywordExtractor):
    keywords = list(extractor.summary.keys())
    selected_keywords = st.multiselect("Filter keywords", keywords)

    for kw, info in extractor.summary.items():
        if selected_keywords and kw not in selected_keywords:
            continue

        st.markdown('____')
        st.markdown(f'#### {kw}')

        for record in info:
            st.markdown(f'* ##### file: {record["fileid"]}')
            st.markdown(f'* ##### score: {record["score"]:.3f}')
            for sent in record['sents']:
                highlighted_sent = sent.replace(kw, f'*__{kw}__*')
                st.markdown(f'> {highlighted_sent}')


@st.cache(hash_funcs={Language: id}, allow_output_mutation=True)
def get_spacy_lang() -> Language:
    return spacy.load('en_core_web_md')


@st.cache(allow_output_mutation=True)
def get_state_store() -> dict:
    """This dictionary is initialized once and can be used to store the state of the application in memory"""

    return {}


if __name__ == "__main__":
    main()
