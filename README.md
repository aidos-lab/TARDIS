# TOAST: Topological Algorithm for Singularity Tracking

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

    $ cd plh
    $ python cli.py ../data/Pinched_torus.txt.gz -q 500 -r 0.05 -R 0.45 -s 0.2 -S 0.6 > ../output/Pinched_torus.txt

This will create a point cloud of 500 sample points with $x, y, z$
coordinates, followed by our Euclidicity score.

### Wedged spheres (with automated parameter selection)

**Warning**: this example might require a long runtime on an ordinary
machine. We ran this on our cluster (see also the [`scripts`](./scripts)
folder in the root directory).

Run the following commands from the root directory of the repository:

    $ cd plh
    $ python cli.py -k 100 -q 2000 -d 2 --num-steps 20 ../data/Wedged_spheres_2D.txt.gz > ../output/Wedged_spheres_2D.txt

This will make use of the automated parameter selection procedure based
on nearest neighbours. Notice that this example uses more query
points; it is of course possible to adjust this parameter.

## License

Our code is released under a BSD-3-Clause license. This license
essentially permits you to freely use our code as desired, integrate it
into your projects, and much more---provided you acknowledge the
original authors. Please refer to [LICENSE.md](./LICENSE.md) for more
information. 

## Issues

This project is maintained by members of the [AIDOS Lab](https://github.com/aidos-lab).
Please open an [issue](https://github.com/aidos-lab/TOAST/issues) in
case you encounter any problems.
