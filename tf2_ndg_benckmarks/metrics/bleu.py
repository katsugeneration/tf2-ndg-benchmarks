"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from tf2_ndg_benckmarks.metrics.utils import ngram
import numpy as np


class Bleu(object):
    """BLEU score calculator."""

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
                len(set(ngram.count_ngram(i, hypothesis).keys()) & set(ngram.count_ngram(i, reference).keys())) /
                sum(ngram.count_ngram(i, hypothesis).values()))
                for i in range(1, n+1)]))

        c = float(len(hypothesis.split(' ')))
        r = float(abs(len(reference.split(' '))))
        BP = 1 if c > r else np.exp(1 - r / c)
        return BP * precision
