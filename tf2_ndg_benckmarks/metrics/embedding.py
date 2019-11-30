"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""


class EmbeddingBase:
    """Embedding based score calculator base."""

    def __init__(self):
        self.emb_path = '/tmp/vector.bin'
