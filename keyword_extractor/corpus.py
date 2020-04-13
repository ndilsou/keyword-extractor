from zipfile import ZipFile

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.data import ZipFilePathPointer
import spacy
from spacy.language import Language
from textacy import Corpus


def create_text_corpus_from_zipfile(zf: ZipFile, pattern='.*\.txt', ensure_loaded=True) -> PlaintextCorpusReader:
    '''
    Loads a text corpus contained in a zipfile.
    '''
    pointer = ZipFilePathPointer(zf)
    corpus = PlaintextCorpusReader(pointer, pattern)

    if ensure_loaded:
        corpus.ensure_loaded()

    return corpus

def create_spacy_corpus(text_corpus: PlaintextCorpusReader, lang: Language) -> Corpus:
    data = ((text_corpus.raw(fid), {'fileid': fid}) for fid in text_corpus.fileids())
    corpus = Corpus(lang, data)
    return corpus

