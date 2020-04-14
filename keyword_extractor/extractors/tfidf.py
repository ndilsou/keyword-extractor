from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn import feature_extraction as fe
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from textacy import Corpus

from .base import KeywordExtractor, ExtractionResult, DocMatch, SentenceMatch


class TfIdfKeywordExtractor(KeywordExtractor):
    '''
    Extract keywords from a corpus using their TfIdf score.
    '''

    _vocab: np.array
    _term_freq_matrix: np.array

    def __init__(self, ngram_range=(1, 1)):
        self._tfidf = fe.text.TfidfVectorizer(stop_words='english', ngram_range=ngram_range)

    def fit(self, corpus: Corpus):
        X = [doc.text for doc in corpus]
        self._term_freq_matrix = self._tfidf.fit_transform(X)

        return self

    def extract(self, corpus: Corpus, topn: int = 5) -> ExtractionResult:
        if sparse.issparse(self._term_freq_matrix):
            self._term_freq_matrix = self._term_freq_matrix.toarray()

        sorted_vocab = sorted(self._tfidf.vocabulary_.items(), key=lambda x: x[-1])
        self._vocab = np.array([tok[0] for tok in sorted_vocab])
        argsorted = np.argsort(self._term_freq_matrix)

        results = defaultdict(list)
        for i, doc in enumerate(corpus):
            top_kw_idx = np.flip(argsorted[i, -topn:])
            keywords = self._vocab[top_kw_idx]
            scores = self._term_freq_matrix[i, top_kw_idx]
            doc_matches = self._extract_doc_matches(corpus.spacy_lang, doc, keywords, scores)

            for kw, match in doc_matches.items():
                results[kw].append(match)

        return ExtractionResult(results)

    def _extract_doc_matches(
            self,
            lang: Language,
            doc: Doc,
            keywords: Sequence[str],
            scores: Sequence[float],
    ) -> Dict[str, DocMatch]:
        '''
        extract and format info for all keywords in a given document.
        attr: str the spacy token attribute to use to match in the sentence search
        '''

        matcher = PhraseMatcher(lang.vocab, attr='LOWER')
        patterns = [lang.make_doc(str(kw)) for kw in keywords]
        matcher.add("Keywords", patterns)
        sents = self._extract_sentence_matches(doc, keywords, matcher, attr='LOWER')

        matches: Dict[str, DocMatch] = {
            kw: DocMatch(doc, kw, score, sents[kw])
            for kw, score
            in zip(keywords, scores)
        }

        return matches
