from typing import Dict, Sequence
from zipfile import ZipFile

from preshed.maps import PreshMap
import pandas as pd
import spacy
from spacy.language import Language
import streamlit as st

from keyword_extractor import extractors
from keyword_extractor.corpus import Corpus, create_spacy_corpus, create_text_corpus_from_zipfile
from keyword_extractor.extractors import KeywordExtractor
from keyword_extractor.utils import summarize_text


def main():
    st.title('Keyword Extractor')
    store = get_state_store()
    sidebar(store)

    if corpus:= store.get('corpus'):
        document_display(store['fileids'], corpus)
        extractor = fit_model(corpus, store['selected_model'])
        keyword_table(extractor)


# @st.cache(hash_funcs={Language: id, PreshMap: id})
def document_display(fileids: Sequence[str], corpus: Corpus):
    # corpus = store['text_corpus']
    # files = corpus.fileids()
    selection = st.multiselect("Select which document to show", fileids)
    # for fileid in selection:
    #     st.subheader(fileid)
    #     if st.checkbox('Full text', key=f'summary-toggle-{fileid}'):
    #         text = corpus.raw(fileid)
    #         st.text(text)
    #     else:
    #         st.text(summarize_text(corpus, fileid))
    # corpus = store['corpus']

    for doc in corpus.get(lambda doc: doc._.meta['fileid'] in selection):
        fileid = doc._.meta['fileid']
        st.subheader(fileid)

        if st.checkbox('Full text', key=f'summary-toggle-{fileid}'):
            # text = corpus.raw(fileid)
            st.text(doc)
        else:
            st.text(summarize_text(doc))


def sidebar(store: dict):
    '''
    A component for all the elements of the sidebar.
    '''

    file_uploader(store)

    if selected_model := st.sidebar.selectbox('Model', ['TfIdf', 'TextRank', 'EmbedRank']):
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
            with st.spinner(text='In progress...'):
                nlp = get_spacy_lang()
            corpus = create_spacy_corpus(text_corpus, nlp)
            store['corpus'] = corpus
    else:
        st.sidebar.info("Upload one or more `.zip` files.")

    if st.sidebar.button("Clear file"):
        store.pop('text_corpus', None)
        store.pop('corpus', None)

    if corpus := store.get('text_corpus'):

        if st.sidebar.checkbox("List documents in corpus?", True):
            st.sidebar.markdown(f'```{corpus.fileids()}```')


def fit_model(corpus: Corpus, selected_model: str, topn: int = 5) -> KeywordExtractor:
    klass = {
        'TfIdf': extractors.TfIdfKeywordExtractor
    }.get(selected_model, None)
    extractor = klass(topn)
    with st.spinner(text='Training in progress...'):
        extractor.fit(corpus)

    return extractor


def keyword_table(extractor: KeywordExtractor):
    st.subheader('Keywords with scores')
    scores = extractor.scores
    df = pd.DataFrame(scores)
    st.dataframe(df)


@st.cache(hash_funcs={Language: id})
def get_spacy_lang() -> Language:
    return spacy.load('en_core_web_md')


@st.cache(allow_output_mutation=True)
def get_state_store() -> dict:
    """This dictionary is initialized once and can be used to store the state of the application in memory"""

    return {}


if __name__ == "__main__":
    main()
