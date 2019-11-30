"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from tf2_ndg_benckmarks.metrics.utils import ngram


class Rouge(object):
    """ROUGE score calculator."""

    def sentence_score(
            self,
            reference: str,
            hypothesis: str,
            mode: str = 'N',
            n: int = 1) -> float:
        """Sentece ROUGE metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.
            mode (str): rouge mode. 'N' or 'L' is supported.
            n (int): n-gram's n.

        Return:
            float: Sentence ROUGE score

        """
        count_ref = ngram.count_ngram(n, reference)
        count_hyp = ngram.count_ngram(n, hypothesis)
        matched = set(count_ref.keys()) & set(count_hyp.keys())

        precision = float(len(matched) / len(count_hyp))
        recall = float(len(matched) / len(count_ref))
        return (precision * recall) / ((1 - 0.5) * precision + 0.5 * recall)
