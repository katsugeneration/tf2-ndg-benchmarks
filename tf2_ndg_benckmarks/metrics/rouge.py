"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from collections import Counter


class Rouge(object):
    """ROUGE score calculator."""

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
            n: int = 1) -> float:
        """Sentece ROUGE metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.
            n (int): n-gram's n.

        Return:
            float: Sentence ROUGE score

        """
        count_ref = self._count_ngram(n, reference)
        count_hyp = self._count_ngram(n, hypothesis)
        matched = set(count_ref.keys()) & set(count_hyp.keys())

        precision = float(len(matched) / len(count_hyp))
        recall = float(len(matched) / len(count_ref))
        return (precision * recall) / ((1 - 0.5) * precision + 0.5 * recall)
