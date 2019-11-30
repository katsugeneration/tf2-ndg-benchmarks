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


class Average(object):
    """BLEU score calculator."""

    def sentence_score(
            self,
            reference: str,
            hypothesis: str) -> float:
        """Sentece Embedding Average metrics.

        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            float: Sentence Embedding Average score

        """
        return 0.0
