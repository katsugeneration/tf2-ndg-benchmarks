"""
Copyright:
    Copyright 2019 by Katsuya SHIMABUKURO.
License:
    MIT, see LICENSE for details.
"""
import pathlib
import gzip
import requests
import tqdm
import numpy as np
from gensim.models import KeyedVectors


FILE_ID = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'
SOURCE_URL = 'https://drive.google.com/uc?export=download&id={file_id}'
SOURCE_URL_WITH_CONFIRM = 'https://drive.google.com/uc?export=download&confirm={code}&id={file_id}'


class EmbeddingBase(object):
    """Embedding based score calculator base."""

    def __init__(self):
        self.emb_path = '/tmp/vector.bin'

        emb_path = pathlib.Path(self.emb_path)
        if emb_path.exists():
            self._load()
            return

        emb_gz_path = pathlib.Path(self.emb_path + '.gz')

        # Downloas Google pre-trained vector bin from Google Drive

        # Get confirmation code
        res = requests.get(SOURCE_URL.format(**{'file_id': FILE_ID}))
        cookies = res.cookies
        res.close()
        code = cookies[next(filter(lambda k: '_warning_' in k, cookies.keys()))]

        # Download file.
        res = requests.get(
                    SOURCE_URL_WITH_CONFIRM.format(**{'file_id': FILE_ID, 'code': code}),
                    cookies=cookies,
                    stream=True)
        pbar = tqdm.tqdm(unit="B", unit_scale=True, desc='Download Google news corpus pre-trained vectors.')
        chunck_size = 1024
        with emb_gz_path.open('wb') as w:
            for chunck in res.iter_content(chunck_size):
                w.write(chunck)
                pbar.update(len(chunck))
        pbar.close()
        res.close()

        # Decompress gzip file.
        with emb_gz_path.open('rb') as f:
            with emb_path.open('wb') as w:
                w.write(gzip.decompress(f.read()))

        self._load()

    def _load(self):
        """Load word2vec model."""
        self.model = KeyedVectors.load_word2vec_format(self.emb_path, binary=True)
        assert 'dog' in self.model

    def _get_vectors_from_sentene(self, sentence):
        """Return contains word vector list."""
        return [self.model.get_vector(w) for w in sentence.split(' ') if w in self.model]

    def _calc_cosine_sim(self, vectors1, vectors2):
        """Calculate cosine similarity."""
        vectors1 /= np.linalg.norm(vectors1, axis=-1)
        vectors2 /= np.linalg.norm(vectors2, axis=-1)
        return np.dot(vectors1, vectors2)


class Average(EmbeddingBase):
    """Embedding based average score calculator."""

    def sentence_score(
            self,
            reference: str,
            hypothesis: str) -> float:
        """Embedding Average metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            float: Embedding Average score

        """
        emb_ref = np.sum(self._get_vectors_from_sentene(reference), axis=0)
        emb_hyp = np.sum(self._get_vectors_from_sentene(hypothesis), axis=0)
        return self._calc_cosine_sim(emb_ref, emb_hyp)


class VectorExtrema(EmbeddingBase):
    """Embedding based vector extrema score calculator."""

    def sentence_score(
            self,
            reference: str,
            hypothesis: str) -> float:
        """Embedding Vector Extrema metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            float: Embedding Vector Extrema score

        """
        def extema(vectors):
            vec_max = np.max(vectors, axis=0)
            vec_min = np.min(vectors, axis=0)
            return list(map(lambda x, y: x if np.abs(x) > np.abs(y) else y, vec_max, vec_min))

        extema_ref = extema(self._get_vectors_from_sentene(reference))
        extema_hyp = extema(self._get_vectors_from_sentene(hypothesis))
        return self._calc_cosine_sim(extema_ref, extema_hyp)


class GreedyMatching(EmbeddingBase):
    """Embedding based greedy matching score calculator."""

    def sentence_score(
            self,
            reference: str,
            hypothesis: str) -> float:
        """Embedding greedy matching metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            float: Embedding Greedy Matching score

        """
        return 0.0
