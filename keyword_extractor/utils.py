import cytoolz as tlz
from spacy.tokens import Doc
from nltk.corpus.reader import PlaintextCorpusReader


def summarize_text(doc: Doc):
    '''
    Reduces a large doc to a few sentences at the beginning middle and end of the document.
    '''
    sentences = list(sent.text for sent in doc.sents)
    doc_start = " ".join(tlz.concat(sentences[:2]))
    mid_i = int(len(sentences)/2)
    doc_mid = " ".join(tlz.concat(sentences[mid_i:mid_i+2]))
    doc_end = " ".join(tlz.concat(sentences[-2:]))

    return f'{doc_start}\n...\n{doc_mid}\n...\n{doc_end}'


def highlight_sentence(sentence, start_char, end_char) -> str:
    '''
    Surrounds the span between start and end by a emphasized markdown.
    '''

    span = sentence[start_char:end_char+1]
    highlighted_sent = sentence[:start_char] + f'*__{span}__*' + sentence[end_char+1:]
    return highlighted_sent
