# tf2-ndg-benchmarks
Neural Dialogue Generation Benchmarks implemented TensorFlow 2.0

![](https://github.com/katsugeneration/tf2-ndg-benchmarks/workflows/build/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tf2-ndg-benchmarks/badge/?version=latest)](https://tf2-ndg-benchmarks.readthedocs.io/en/latest/?badge=latest)

# Usage

Run under the command for setup.

```sh
poetry install
```

Run under python code for using metrics.

```python
from tf2-ndg-benchmarks import metrics

hypothesis = 'It is a guide to action that ensures that the military will forever heed Party commands'
reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
bleu = metrics.Bleu()
score = bleu.sentence_score(reference, hypothesis)
```
