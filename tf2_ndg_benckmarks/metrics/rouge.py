"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""


class Rouge(object):
    """ROUGE score calculator."""

    def sentence_score(
            self,
            reference: str,
            hypothesis: str) -> float:
        """Sentece BLEU metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            float: Sentence BLEU score

        """
        return 0.0
