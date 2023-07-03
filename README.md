# TARDIS: Topological Algorithms for Robust DIscovery of Singularities

[![arXiv](https://img.shields.io/badge/arXiv-2210.00069-b31b1b.svg)](https://arxiv.org/abs/2210.00069) [![Maintainability](https://api.codeclimate.com/v1/badges/4656850a9d0eb2f85b6e/maintainability)](https://codeclimate.com/github/aidos-lab/TARDIS/maintainability) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/TARDIS) ![GitHub](https://img.shields.io/github/license/aidos-lab/TARDIS)

![TARDIS icon](./TARDIS.svg)

This is the code for our preprint on singularity analysis:

```bibtex
@Unpublished{vonRohrscheidt22a,
  title         = {Topological Singularity Detection at Multiple Scales},
  author        = {von Rohrscheidt, Julius and Rieck, Bastian},
  year          = 2022,
  archiveprefix = {arXiv},
  eprint        = {2210.00069},
  primaryclass  = {cs.LG},
  abstract      = {The manifold hypothesis, which assumes that data lie on or close to an unknown manifold of low intrinsic dimensionality, is a staple of modern machine learning research. However, recent work has shown that real-world data exhibit distinct non-manifold structures, which result in singularities that can lead to erroneous conclusions about the data. Detecting such singularities is therefore crucial as a precursor to interpolation and inference tasks. We address detecting singularities by developing (i) persistent local homology, a new topology-driven framework for quantifying the intrinsic dimension of a data set locally, and (ii) Euclidicity, a topology-based multi-scale measure for assessing the 'manifoldness' of individual points. We show that our approach can reliably identify singularities of complex spaces, while also capturing singular structures in real-world data sets.},
  type          = {Preprint},
  repository    = {https://github.com/aidos-lab/TARDIS},
}
```

## Installation

Our code has been tested with Python 3.8 and Python 3.9 under Mac OS
X and Linux. Other Python versions *may* not support all dependencies.

The recommended way to install the project is via [`poetry`](https://python-poetry.org/).
If this is available, installation should work very quickly:

    $ poetry install

Recent versions of `pip` should also be capable of installing the
project directly:

    $ pip install .

## Experiments

To reproduce the main experiments in our paper, we ship synthetic data
sets in the repository and offer the automated capability to download
the computer vision data sets&nbsp;(`MNIST` and `FashionMNIST`). For
reasons of simplicity, we suggest to reproduce the experiments with
synthetic point clouds first as they run quickly even on a standard
desktop computer.

All experiments make use of the script `cli.py`, which provides
a command-line interface to our framework. Given input parameters for
the local annuli, this script will calculate Euclidicity values as
described in the paper. For reasons of simplicity, all output is
provided to `stdout`, i.e. the standard output of your terminal, and
needs to be redirected to a file for subsequent analysis.

We will subsequently provide the precise commands to reproduce the
experiments; readers are invited to take a look at the code in `cli.py`
or call `python cli.py --help` in order to see what additional options
are available for processing data.

### Pinched torus

Run the following commands from the root directory of the repository:

    $ cd tardis
    $ python cli.py ../data/Pinched_torus.txt.gz -q 500 -r 0.05 -R 0.45 -s 0.2 -S 0.6 > ../output/Pinched_torus.txt

This will create a point cloud of 500 sample points with $x, y, z$
coordinates, followed by our Euclidicity score.

### Wedged spheres (with automated parameter selection)

**Warning**: this example might require a long runtime on an ordinary
machine. We ran this on our cluster (see also the [`scripts`](./scripts)
folder in the root directory).

Run the following commands from the root directory of the repository:

    $ cd tardis
    $ python cli.py -k 100 -q 2000 -d 2 --num-steps 20 ../data/Wedged_spheres_2D.txt.gz > ../output/Wedged_spheres_2D.txt

This will make use of the automated parameter selection procedure based
on nearest neighbours. Notice that this example uses more query
points; it is of course possible to adjust this parameter.

## API & examples

Check out the [examples folder](/examples) for some code snippets that
demonstrate how to use TARDIS in your own code. They all make use of the
[preliminary API](/tardis/api.py).

## License

Our code is released under a BSD-3-Clause license. This license
essentially permits you to freely use our code as desired, integrate it
into your projects, and much more---provided you acknowledge the
original authors. Please refer to [LICENSE.md](./LICENSE.md) for more
information. 

## Issues

This project is maintained by members of the [AIDOS Lab](https://github.com/aidos-lab).
Please open an [issue](https://github.com/aidos-lab/TARDIS/issues) in
case you encounter any problems.
