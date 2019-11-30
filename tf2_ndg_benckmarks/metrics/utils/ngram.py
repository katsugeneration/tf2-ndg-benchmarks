"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from collections import Counter


def count_ngram(
        n: int,
        target: str) -> Counter:
    """Count n-gram.

    Args:
        n (int): n-gram's n.
        target: (str): target sentence.

    Return:
        Counter: python counter object.

    """
    tokens = target.split(' ')
    N = len(tokens)
    counter = Counter(map(lambda x: ' '.join(x), zip(*[tokens[i:N-n+i+1] for i in range(n)])))
    return counter
