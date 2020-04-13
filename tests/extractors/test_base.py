from keyword_extractor.extractors.base import extract_doc_info, extract_parent_sentences

def test_extract_parent_sentences(nlp, simple_doc):
    keywords = ['hope']
    expected = {'hope': ['In the face of despair, you believe there can be hope.']}
    actual = extract_parent_sentences(nlp, simple_doc, keywords)
    assert actual == expected
