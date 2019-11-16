"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from collections import Counter
import numpy as np


class Bleu(object):
    """BLEU score calculator."""

    def _count_ngram(
            self,
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

    def sentence_score(
            self,
            reference: str,
            hypothesis: str,
            n: int = 4) -> float:
        """Sentece BLEU metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.
            n (int): n-gram's n.

        Return:
            float: Sentence BLEU score

        """
        precision = np.exp(sum(
            [1./float(n) * np.log(
                len(set(self._count_ngram(i, hypothesis).keys()) & set(self._count_ngram(i, reference).keys())) /
                sum(self._count_ngram(i, hypothesis).values()))
                for i in range(1, n+1)]))

        c = float(len(hypothesis.split(' ')))
        r = float(abs(len(reference.split(' '))))
        BP = 1 if c > r else np.exp(1 - r / c)
        return BP * precision
