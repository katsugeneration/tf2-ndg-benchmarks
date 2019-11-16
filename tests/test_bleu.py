"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from nose.tools import eq_
import nltk
from tf2_ndg_benckmarks.metrics import bleu


class TestBleu:
    def test_sentence_score(self):
        reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
        hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        scorer = bleu.Bleu()

        expect = nltk.translate.bleu_score.sentence_bleu([reference.split(' ')], hypothesis.split(' '))
        eq_(expect, scorer.sentence_score(reference, hypothesis))

    def test_sentence_score_bleu_2(self):
        reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
        hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        scorer = bleu.Bleu()

        expect = nltk.translate.bleu_score.sentence_bleu([reference.split(' ')], hypothesis.split(' '), weights=(0.5, 0.5))
        eq_(expect, scorer.sentence_score(reference, hypothesis, n=2))
