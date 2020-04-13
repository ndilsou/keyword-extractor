from __future__ import annotations
from abc import abstractmethod
from typing import Dict, List, Sequence, Tuple, TypedDict
from dataclasses import dataclass, field

from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from textacy import Corpus


class KeywordExtractor:
    '''
    Extract keywords from a corpus.
    '''

    def __init__(self, topn: int = 5):
        self.topn = topn

    @abstractmethod
    def fit(self, corpus: Corpus):
        raise NotImplementedError

    @abstractmethod
    def extract(self, corpus: Corpus) -> ExtractionResult:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    def _extract_sentence_matches(
            doc: Doc,
            keywords: Sequence[str],
            matcher,
            attr: str
    ) -> Dict[str, List[SentenceMatch]]:
        matches = matcher(doc)
        results = {kw: [] for kw in keywords}

        for match_id, start, end in matches:
            span = doc[start:end]
            sent = span.sent
            key = _get_key(span, attr)
            start_char = span.start_char - span.sent.start_char
            end_char = span.end_char - span.sent.start_char
            print(key)
            results[key].append(SentenceMatch(sent, start_char, end_char))

        return results


class ExtractionResult:
    raw: Dict[str, List[DocMatch]]

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def keywords(self) -> List[str]:
        '''
        list of keywords extracted from the document as strings
        '''

        return list(self.raw.keys())

    @property
    def scores(self) -> Dict[str, Dict[str, float]]:
        '''
        list of keywords with their scores per document.
        '''

        return {
            kw: {
                doc_match.fileid: doc_match.score

                for doc_match in matches
            }

            for kw, matches in self.raw.items()
        }


@dataclass
class DocMatch:
    '''
    Captures the details of a keyword match in a given document.
    '''
    doc: Doc
    keyword: str
    score: float
    sents: Sequence[SentenceMatch]
    # fileid: str

    @property
    def fileid(self) -> str:
        return self.doc._.meta['fileid']


@dataclass
class SentenceMatch:
    sent: Span
    start_char: int
    end_char: int
    # text: str = field(init=False)

    @property
    def text(self) -> str:
        return self.sent.text

# def extract_doc_matches(
#         lang: Language,
#         doc: Doc,
#         keywords: Sequence[str],
#         scores: Sequence[float],
# ) -> Dict[str, DocMatch]:
#     '''
#     extract and format info for all keywords in a given document.
#     attr: str the spacy token attribute to use to match in the sentence search
#     '''

#     matcher = Matcher(lang.vocab)
#     patterns = [[{'LEMMA': tok} for tok in kw.split()] for kw in keywords]
#     matcher.add("Keywords", patterns)
#     sents = extract_sentence_matches(doc, keywords, matcher, attr='LEMMA')

#     matches: Dict[str, DocMatch] = {}

#     for kw, score in zip(keywords, scores):
#         # info[kw] = {
#         #     'fileid': doc._.meta['fileid'],
#         #     'score': score,
#         #     'sents': sents[kw]
#         # }
#         matches[kw] = DocMatch(doc, kw, score, sents['kw'])

#     return matches

# def extract_sentence_matches(
#         doc: Doc,
#         keywords: Sequence[str],
#         matcher,
#         attr: str
# ) -> Dict[str, List[SentenceMatch]]:
#     matches = matcher(doc)
#     results = {kw: [] for kw in keywords}

#     for match_id, start, end in matches:
#         span = doc[start:end]
#         sent = span.sent.text
#         key = _get_key(span, attr)
#         start_char = span.start_char - span.sent.start_char
#         end_char = span.end_char - span.sent.start_char
#         # results[key].append({'sent': sent, 'start_char': start_char, 'end_char': end_char})
#         results[key].append(SentenceMatch(sent, start_char, end_char))

#     return results


def _get_key(span, attr):
    attr = attr.upper()
    key: str
    if attr == 'LOWER':
        key = span.text.lower()
    elif attr == 'LEMMA':
        key = span.lemma_
    else:
        key = span.text
    return key
