altx — Adaptive Law-Based Transformation
=========================================

.. image:: https://github.com/halmosb/altx/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/halmosb/altx/actions/workflows/tests.yml
   :alt: Python tests

.. image:: ../../.badges/coverage.svg
   :target: https://github.com/halmosb/altx/actions/workflows/tests.yml
   :alt: Coverage

**altx** is an open-source Python package for efficient time series
classification (TSC). It converts raw time series into a linearly
separable feature space by discovering *linear laws* — eigenvectors of
symmetric Hankel matrices that correspond to conserved quantities in the
data. Variable-length shifted time windows let the algorithm capture
patterns at multiple temporal scales simultaneously.

``altx`` achieves state-of-the-art performance on TSC benchmarks from
physics and related domains while keeping computational overhead low.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/modules
   development

----

How it works
------------

The pipeline has two phases:

**Train** — For each ``(R, L, K)`` configuration, every training instance
is scanned with a sliding window of length ``R`` (step ``K``). Each
window is embedded into a symmetric ``L×L`` Hankel matrix, and the
eigenvector for the smallest absolute eigenvalue is stored as a *law*.

**Transform** — For a test instance, windows are projected onto every
stored law. The resulting scores are partitioned by training class and
summarised into scalar features by a chosen extraction method. The final
feature vector has length ``len(RLK) × noc × n_methods × m``.

Quick start
-----------

.. code-block:: python

   import torch
   from altx import ALT as Altx

   # Training data: 6 univariate instances of length 50, two classes
   _ = torch.manual_seed(0)
   train_data    = torch.randn(6, 50)
   train_classes = torch.tensor([0, 0, 0, 1, 1, 1])

   model = Altx(train_data, train_classes, L=3, K=1)
   model.train()

   # Transform a test set
   test_data = torch.randn(4, 50)
   features  = model.transform_set(test_data, [["mean", 0.05]])
   print(features.shape)   # torch.Size([4, 2])

See the :doc:`usage` page for a full walkthrough with all options and
extraction methods.

Citation
--------

If you use ``altx`` in your research, please cite:

| [1] M. T. Kurbucz, B. Hajós, B. P. Halmos, V. Á. Molnár, A. Jakovác,
|     *Adaptive law-based feature representation for time series classification*,
|     Scientific Reports **15** (1), 41775, 2025.
|     `https://doi.org/10.1038/s41598-025-25667-0 <https://doi.org/10.1038/s41598-025-25667-0>`_

| [2] B. P. Halmos, B. Hajós, V. Á. Molnár, M. T. Kurbucz, A. Jakovác,
|     *altx: a Python package for adaptive law-based transformation in time series classification*,
|     Machine Learning: Science and Technology **7** (1), 015034, 2026.
|     `https://doi.org/10.1088/2632-2153/ae3e4f <https://doi.org/10.1088/2632-2153/ae3e4f>`_

.. code-block:: bibtex

   @article{kurbucz2025adaptive,
     title   = {Adaptive law-based feature representation for time series classification},
     author  = {Kurbucz, Marcell T and Haj{\'o}s, Bal{\'a}zs and Halmos, Bal{\'a}zs P
                and Moln{\'a}r, Vince {\'A} and Jakov{\'a}c, Antal},
     journal = {Scientific Reports},
     volume  = {15},
     number  = {1},
     pages   = {41775},
     year    = {2025},
     doi     = {10.1038/s41598-025-25667-0},
   }

   @article{halmos2026altx,
     title   = {altx: a python package for adaptive law-based transformation
                in time series classification},
     author  = {Halmos, Bal{\'a}zs P and Haj{\'o}s, Bal{\'a}zs and Moln{\'a}r, Vince {\'A}
                and Kurbucz, Marcell T and Jakov{\'a}c, Antal},
     journal = {Machine Learning: Science and Technology},
     volume  = {7},
     number  = {1},
     pages   = {015034},
     year    = {2026},
     doi     = {10.1088/2632-2153/ae3e4f},
   }

License
-------

``altx`` is released under the **GPLv3** license.
