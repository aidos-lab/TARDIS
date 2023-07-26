TARDIS: Topological Algorithms for Robust DIscovery of Singularities
====================================================================

The manifold hypothesis drives most of modern machine learning research,
but what if you are **not** dealing with a manifold but a more complicated space?
TARDIS uses a topology-driven approach to identify singularities in
high-dimensional data sets at multiple scales, giving you a better
overview of what is in your data.

How can TARDIS help you?
------------------------

* Find out whether your data set contains singular regions, i.e. regions
  that are not adequately described by Euclidean space.

* Discover whether dimensionality reduction algorithms are embedding
  your data correctly or resulting in distortion.

* Assess the overall complexity of your data set in an unsupervised
  fashion.

Interested?
-----------

Read more about TARDIS in our `ICML paper <https://proceedings.mlr.press/v202/von-rohrscheidt23a.html>`_
and consider citing us:

.. code-block:: bibtex

  @inproceedings{vonRohrscheidt23a,
    title       = {Topological Singularity Detection at Multiple Scales},
    author      = {von Rohrscheidt, Julius and Rieck, Bastian},
    year        = 2023,
    booktitle   = {Proceedings of the 40th International Conference on Machine Learning},
    publisher   = {PMLR},
    series      = {Proceedings of Machine Learning Research},
    number      = 202,
    pages       = {35175--35197},
    editor      = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
    abstract    = {The manifold hypothesis, which assumes that data lies on or close to an unknown manifold of low intrinsic dimension, is a staple of modern machine learning research. However, recent work has shown that real-world data exhibits distinct non-manifold structures, i.e. singularities, that can lead to erroneous findings. Detecting such singularities is therefore crucial as a precursor to interpolation and inference tasks. We address this issue by developing a topological framework that (i) quantifies the local intrinsic dimension, and (ii) yields a Euclidicity score for assessing the `manifoldness' of a point along multiple scales. Our approach identifies singularities of complex spaces, while also capturing singular structures and local geometric complexity in image data.}
  }

Documentation
-------------

.. toctree::
    :maxdepth: 2
    :caption: Contents:

.. toctree::
    :maxdepth: 2
    :caption: Modules

    api
    data
    utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
