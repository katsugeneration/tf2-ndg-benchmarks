"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from tf2_ndg_benckmarks.metrics.utils import ngram
from typing import List, Set


class Distinct(object):
    """DISTINCT score calculator."""

    def sentence_score(
            self,
            hypotheses: List[str],
            n: int = 1) -> float:
        """Sentece DISTINCT metrics.

        Args:
            hypotheses: (List[str]): hypothesis sentence.
            n (int): n-gram's n.

        Return:
            float: Sentence DISTINCT score

        """
        count_hypotheses = []
        for h in hypotheses:
            count_hypotheses.append(ngram.count_ngram(n, h))

        distinct: Set[str] = set()
        for ch in count_hypotheses:
            distinct |= set(ch.keys())
        denom = sum(map(lambda v: sum(v.values()), count_hypotheses))
        return float(len(distinct) / denom)
