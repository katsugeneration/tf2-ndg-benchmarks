"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from typing import List


class Distinct(object):
    """DISTINCT score calculator."""

    def sentence_score(
            self,
            hypotheses: List[str]) -> float:
        """Sentece DISTINCT metrics.

        Args:
            hypotheses: (List[str]): hypothesis sentence.

        Return:
            float: Sentence DISTINCT score

        """
        return 0.0
