<!---
   tf2-ndg-benchmarks documentation master file, created by
   sphinx-quickstart on Mon Nov 11 10:07:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
--->

# Welcome to tf2-ndg-benchmarks's documentation!

```eval_rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:
```

# Usage

Run under the command for setup.

```sh
poetry install
```

Run under python code for using metrics.

```python
from tf2-ndg-benchmarks import metrics

reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
bleu = metrics.bleu.Bleu()
score = bleu.sentence_score(reference, hypothesis)
```


# Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```