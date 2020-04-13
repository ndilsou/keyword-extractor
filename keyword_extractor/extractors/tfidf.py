from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn import feature_extraction as fe
from textacy import Corpus

from .base import KeywordExtractor, KeywordInfo, extract_doc_info


class TfIdfKeywordExtractor(KeywordExtractor):
    '''
    Extract keywords from a corpus using their TfIdf score.
    '''

    summary: Dict[str, List[KeywordInfo]]
    _vocab: np.array
    _term_freq_matrix: np.array

    def __init__(self, topn: int = 5):
        super().__init__(topn)
        self.summary = defaultdict(list)
        self._tfidf = fe.text.TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

    def fit(self, corpus: Corpus):
        X = [doc.text for doc in corpus]
        self._term_freq_matrix = self._tfidf.fit_transform(X)

        if sparse.issparse(self._term_freq_matrix):
            self._term_freq_matrix = self._term_freq_matrix.toarray()

        sorted_vocab = sorted(self._tfidf.vocabulary_.items(), key=lambda x: x[-1])
        self._vocab = np.array([tok[0] for tok in sorted_vocab])
        argsorted = np.argsort(self._term_freq_matrix)

        for i, doc in enumerate(corpus):
            top_kw_idx = np.flip(argsorted[i, -self.topn:])
            keywords = self._vocab[top_kw_idx]
            scores = self._term_freq_matrix[i, top_kw_idx]
            doc_info = extract_doc_info(corpus.spacy_lang, doc, keywords, scores, attr='TEXT')

            for kw, info in doc_info.items():
                self.summary[kw].append(info)

        return self

    @property
    def keywords(self) -> List[str]:
        '''
        list of keywords extracted from the document as strings
        '''

        return list(self.summary.keys())

    @property
    def scores(self) -> Dict[str, Dict[str, float]]:
        '''
        list of keywords with their scores per document.
        '''

        return {
            kw: {
                record['fileid']: record['score']

                for record in info
            }

            for kw, info in self.summary.items()
        }

    @property
    def fileids(self) -> List[Tuple[str, str]]:
        '''
        list of keywords with their parent fileid.
        '''

    @property
    def sents(self) -> List[Tuple[str, str, str]]:
        '''
        flat list of keywords witht their parent fileid and sentence.
        '''
