"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from nose.tools import ok_, eq_
import nltk
from tf2_ndg_benckmarks.metrics import distinct


class TestDistinct:
    def test_sentence_score(self):
        hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        hypothesis2 = 'It is a guide to action that ensures that the military will forever heed Party commands'
        hypothesis3 = 'It is fine day.'
        scorer = distinct.Distinct()

        eq_(0.5, scorer.sentence_score([hypothesis3, hypothesis3]))
        ok_(scorer.sentence_score([hypothesis1, hypothesis1]) < scorer.sentence_score([hypothesis1, hypothesis2]))
