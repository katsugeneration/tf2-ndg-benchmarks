"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
from nose.tools import ok_
import os
from tf2_ndg_benckmarks.metrics import embedding


def test_load():
    base = embedding.EmbeddingBase()
    ok_(os.path.exists(base.emb_path))
