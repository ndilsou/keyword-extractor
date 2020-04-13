from abc import abstractmethod
from typing import Dict, List, Sequence, Tuple, TypedDict

from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
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

    @property
    def keywords(self) -> List[str]:
        '''
        list of keywords extracted from the document as strings
        '''
        raise NotImplementedError

    @property
    def scores(self) -> Dict[str, Dict[str, float]]:
        '''
        list of keywords with their scores per document.
        '''
        raise NotImplementedError

    @property
    def fileids(self) -> List[Tuple[str, str]]:
        '''
        list of keywords with their parent fileid.
        '''
        raise NotImplementedError

    @property
    def sents(self) -> List[Tuple[str, str, str]]:
        '''
        flat list of keywords witht their parent fileid and sentence.
        '''
        raise NotImplementedError


class KeywordInfo(TypedDict):
    fileid: str
    score: float
    sents: Sequence[str]


def extract_doc_info(
        lang: Language,
        doc: Doc,
        keywords: Sequence[str],
        scores: Sequence[float],
        attr: str = 'TEXT'
) -> Dict[str, KeywordInfo]:
    '''
    extract and format info for all keywords in a given document.
    attr: str the spacy token attribute to use to match in the sentence search
    '''
    sents = extract_parent_sentences(lang, doc, keywords, attr)

    info: Dict[str, KeywordInfo] = {}

    for kw, score in zip(keywords, scores):
        info[kw] = {
            'fileid': doc._.meta['fileid'],
            'score': score,
            'sents': sents[kw]
        }

    return info


def extract_parent_sentences(nlp: Language, doc: Doc, keywords, attr: str = 'TEXT'):
    matcher = Matcher(nlp.vocab)
    patterns = [[{attr: tok} for tok in kw.split()] for kw in keywords]
    matcher.add("Keywords", patterns)
    matches = matcher(doc)
    results = {kw: [] for kw in keywords}

    for match_id, start, end in matches:
        span = doc[start:end]
        sent = span.sent.text
        k = span.lemma_ if attr.lower() == 'LEMMA' else span.text
        results[k].append(sent)

    return results
