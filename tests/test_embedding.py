"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from nose.tools import ok_
import os
import sys
from tf2_ndg_benckmarks.metrics import embedding


class TestBase:
    def test_load(self):
        base = embedding.EmbeddingBase(emb_path='./vector.bin')
        ok_(os.path.exists(base.emb_path))
        ok_(base.model)


class TestAverage:
    def test_sentence_score(self):
        reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
        hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        scorer = embedding.Average(emb_path='./vector.bin')

        ok_(scorer.sentence_score(reference, reference) - 1.0 <= 0.000001)
        ok_(scorer.sentence_score(reference, hypothesis) > 0.0)


class TestVectorExtrema:
    def test_sentence_score(self):
        reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
        hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        scorer = embedding.VectorExtrema(emb_path='./vector.bin')

        ok_(scorer.sentence_score(reference, reference) - 1.0 <= 0.000001)
        ok_(scorer.sentence_score(reference, hypothesis) > 0.0)


class TestGreedyMatching:
    def test_sentence_score(self):
        reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
        hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        scorer = embedding.GreedyMatching(emb_path='./vector.bin')

        ok_(scorer.sentence_score(reference, reference) - 1.0 <= 0.000001)
        ok_(scorer.sentence_score(reference, hypothesis) > 0.0)
