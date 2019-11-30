"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from nose.tools import eq_
from sumeval.metrics.rouge import RougeCalculator
from tf2_ndg_benckmarks.metrics import rouge


class TestRouge:
    def test_sentence_score(self):
        reference = 'I went to Mars'
        hypothesis = 'I went to the Mars from my living town.'
        scorer = rouge.Rouge()

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        expect = rouge_calc.rouge_n(hypothesis, reference, n=1)
        eq_(expect, scorer.sentence_score(reference, hypothesis))

    def test_sentence_score_N_2(self):
        reference = 'I went to Mars'
        hypothesis = 'I went to the Mars from my living town.'
        scorer = rouge.Rouge()

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        expect = rouge_calc.rouge_n(hypothesis, reference, n=2)
        eq_(expect, scorer.sentence_score(reference, hypothesis, n=2))

    def test_sentence_score_L(self):
        reference = 'It\'s my living town'
        hypothesis = 'I went to the Mars from my living town.'
        scorer = rouge.Rouge()

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        expect = rouge_calc.rouge_l(hypothesis, reference)
        eq_(expect, scorer.sentence_score(reference, hypothesis, mode='L'))
