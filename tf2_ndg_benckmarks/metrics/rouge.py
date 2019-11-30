"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from collections import Counter


class Rouge(object):
    """ROUGE score calculator."""

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
        count_ref = Counter(reference.split(' '))
        count_hyp = Counter(hypothesis.split(' '))
        matched = set(count_ref.keys()) & set(count_hyp.keys())

        precision = float(len(matched) / len(count_hyp))
        recall = float(len(matched) / len(count_ref))
        return (precision * recall) / ((1 - 0.5) * precision + 0.5 * recall)
