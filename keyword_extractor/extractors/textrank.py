from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn import feature_extraction as fe
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from textacy import Corpus
from textacy.ke.textrank import textrank

from .base import KeywordExtractor, ExtractionResult, DocMatch, SentenceMatch


class TextRankKeywordExtractor(KeywordExtractor):
    '''
    Extract keywords from a corpus using TextRank.
    '''

    include_pos: Tuple[str]
    window_size: int
    edge_weighting: str

    def __init__(
            self,
            topn: int = 5,
            window_size: int = 3,
            include_pos=('NOUN', 'PROPN', 'ADJ'),
            edge_weighting: str = 'binary'
    ):
        super().__init__(topn)
        self.include_pos = include_pos
        self.window_size = window_size
        self.edge_weighting = edge_weighting

    def fit(self, corpus: Corpus):
        del corpus
        return self

    def extract(self, corpus: Corpus) -> ExtractionResult:
        results = defaultdict(list)
        for i, doc in enumerate(corpus):
            keyterms = textrank(doc,
                                edge_weighting=self.edge_weighting,
                                include_pos=self.include_pos,
                                topn=self.topn,
                                window_size=self.window_size)

            keywords, scores = list(zip(*keyterms))

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

        matcher = Matcher(lang.vocab)
        patterns = [[{'LEMMA': tok} for tok in kw.split()] for kw in keywords]
        matcher.add("Keywords", patterns)
        sents = self._extract_sentence_matches(doc, keywords, matcher, attr='LEMMA')

        matches: Dict[str, DocMatch] = {
            kw: DocMatch(doc, kw, score, sents[kw])
            for kw, score
            in zip(keywords, scores)
        }

        return matches
