
class Bleu(object):
    """BLEU score calculator."""

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
