import numpy as np

def test_argsort_routine():
    topn = 2
    vocab = np.array(['hello', 'are', 'you', 'how', 'who'])
    term_freq_matrix = np.array([[2, 0.1, 1, 0.2, 0.21],
                                 [0.1, 0.2, 3, 0.6, 2.1]])
    expected_1 = ['hello', 'you']
    expected_2 = ['you', 'who']

    argsorted = np.argsort(term_freq_matrix)

    actual_1 = vocab[np.flip(argsorted[0, -topn:])]
    assert np.array_equal(actual_1, expected_1)
    actual_2 = vocab[np.flip(argsorted[1, -topn:])]
    assert np.array_equal(actual_2, expected_2)

