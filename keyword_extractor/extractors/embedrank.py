from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn import feature_extraction as fe
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from textacy import Corpus
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_hub as hub

from .base import KeywordExtractor, ExtractionResult, DocMatch, SentenceMatch

MODEL_URLS = {
    'Transformer': 'https://tfhub.dev/google/universal-sentence-encoder-large/5',
    'DAN': 'https://tfhub.dev/google/universal-sentence-encoder/4'
}


class EmbedRankKeywordExtractor(KeywordExtractor):
    '''
    Extract keywords from a corpus using the EmbedRank or EmbedRank++ approach with embedding from Universal Sentence Encoder.
    references:
    * "Simple Unsupervised Keyphrase Extraction using Sentence Embeddings" @ https://arxiv.org/abs/1801.04470
    * "Universal Sentence Encoder" @ https://arxiv.org/abs/1803.11175
    '''

    MODEL_URLS = {
        'Transformer': 'https://tfhub.dev/google/universal-sentence-encoder-large/5',
        'DAN': 'https://tfhub.dev/google/universal-sentence-encoder/4'
    }

    def __init__(self, variant='DAN'):
        self.variant = variant

    def fit(self, corpus: Corpus):
        del corpus
        self.encoder = hub.load(self.MODEL_URLS[self.variant])

        return self

    def extract(self, corpus: Corpus, topn: int = 5) -> ExtractionResult:
        results = defaultdict(list)
        # TODO: Trivially parallelizable. need to wrap around a ProcessPoolExecutor for large enough corpora.
        for doc in corpus:
            doc_embedding = self.encoder([doc.text]).numpy()

            keywords, scores = self._get_top_keywords_for_doc(doc, topn)
            doc_matches = self._extract_doc_matches(corpus.spacy_lang, doc, keywords, scores)

            for kw, match in doc_matches.items():
                results[kw].append(match)

        return ExtractionResult(results)

    def _get_top_keywords_for_doc(self, doc: Doc, topn: int) -> Tuple[List[str], List[float]]:
        doc_embedding = self.encoder([doc.text]).numpy()

        noun_chunks = list({nc.text for nc in doc.noun_chunks})
        candidate_embeddings = self.encoder(noun_chunks).numpy()
        similiarities = cosine_similarity(doc_embedding, candidate_embeddings)
        argsorted = np.argsort(similiarities)
        top_kw_idx = np.flip(argsorted[-topn:])
        scores = similiarities[top_kw_idx]
        keywords = [noun_chunks[i] for i in top_kw_idx]
        return keywords, scores

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


        matcher = PhraseMatcher(lang.vocab, attr='TEXT')
        patterns = [lang.make_doc(str(kw)) for kw in keywords]
        matcher.add("Keywords", patterns)
        sents = self._extract_sentence_matches(doc, keywords, matcher, attr='TEXT')

        matches: Dict[str, DocMatch] = {
            kw: DocMatch(doc, kw, score, sents[kw])
            for kw, score
            in zip(keywords, scores)
        }

        return matches


def cosine_similarity(a1, a2):
    return a1.dot(a2.T).flatten()


def get_marginal_relevance(candidate_embeddings, doc_embedding, l=0.5):
    similiarities = cosine_similarity(doc_embedding, candidate_embeddings)
    scaled_similarities = min_max_scale(similiarities)
    norm_similarities = normalize_importances(scaled_similarities)

    candidate_similarities = candidate_embeddings.dot(candidate_embeddings.T)
    candidate_similarities = candidate_similarities - np.diag(candidate_embeddings) * np.eye(candidate_embeddings.shape[0])
    scaled_candidate_similarities = min_max_scale(candidate_similarities)
    norm_candidate_similarities = normalize_importances(scaled_candidate_similarities)

    max_sim_by_candidate = norm_candidate_similarities.max(axis=0)

    marginal_relevance = l * norm_similarities - (1 - l) * max_sim_by_candidate
    return marginal_relevance


def normalize_importances(importances):
    return 0.5 + (importances - importances.mean()) / importances.std()


def min_max_scale(a):
    return (a - a.min()) / (a.max() - a.min())
