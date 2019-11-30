"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from tf2_ndg_benckmarks.metrics.utils import ngram


class Rouge(object):
    """ROUGE score calculator."""

    def _calc_lcs(
            self,
            reference: str,
            hypothesis: str) -> int:
        """Calculate Longest Common Subsequence between sentences.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            int: Longest Common Subsequence length.

        """
        words_ref = ['<S>'] + reference.split(' ')
        words_hyp = ['<S>'] + hypothesis.split(' ')

        # calc LCS by DP
        N = len(words_ref)
        M = len(words_hyp)
        LCS = []

        for i in range(N):
            LCS.append([0] * M)

        for i in range(1, N):
            for j in range(1, M):
                if words_ref[i - 1] == words_hyp[j - 1]:
                    LCS[i][j] = LCS[i - 1][j - 1] + 1
                else:
                    LCS[i][j] = max(LCS[i - 1][j], LCS[i][j - 1])

        return LCS[N-1][M-1]

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
        if mode == 'N':
            count_ref = ngram.count_ngram(n, reference)
            count_hyp = ngram.count_ngram(n, hypothesis)
            matched = set(count_ref.keys()) & set(count_hyp.keys())
            precision = float(len(matched) / len(count_hyp))
            recall = float(len(matched) / len(count_ref))

        elif mode == 'L':
            lcs = self._calc_lcs(reference, hypothesis)
            precision = float(lcs / len(reference.split(' ')))
            recall = float(lcs / len(hypothesis.split(' ')))

        else:
            raise ValueError('Rouge mode %s is not supported. Please set N or L.' % mode)

        return (precision * recall) / ((1 - 0.5) * precision + 0.5 * recall)
