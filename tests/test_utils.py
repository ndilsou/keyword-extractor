from keyword_extractor.utils import highlight_sentence

def test_highlight_sentence():
    sent = 'my name is John'
    expected_1 = 'my name is *__John__*'
    assert highlight_sentence(sent, 11, 14) == expected_1

    expected_2 = 'my *__name__* is John'
    assert highlight_sentence(sent, 3, 6) == expected_2

    expected_3 = '*__my__* name is John'
    assert highlight_sentence(sent, 0, 1) == expected_3
