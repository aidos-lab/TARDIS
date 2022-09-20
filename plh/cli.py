"""Command-line interface for Euclidicity calculations.

This script is the main command-line interface for our Euclidicity
calculations. It supports loading various input formats, for which
it will calculate Euclidicity scores.
"""

import argparse
import joblib
import os

import numpy as np

from plh.euclidicity import Euclidicity


def load(filename):
    """Load data from filename, depending on input type."""
    ext = os.path.splitext(filename)[1]
    if ext == ".txt":
        return np.loadtxt(filename)
    elif ext == ".npz":
        return np.load(filename)["data"]

    return None


def estimate_scales(X, k_max):
    """Perform simple scale estimation of the data set.

    Parameters
    ----------
    k_max : int
        Maximum number of neighbours to consider for the local scale
        estimation.

    Returns
    --------
    Tuple
        A tuple consisting of the minimum and maximum inner and outer
        radius, respectively.
    """
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    distances, _ = tree.query(X, k=k_max, return_distance=True)

    # Ignore the distance to ourself, as we know that one already.
    distances = distances[:, 1:]

    # TODO: We could pick something smarter here...
    r_min = np.mean(distances[:, 0])
    r_max = np.mean(distances[:, 5])
    s_min = np.mean(distances[:, 5])
    s_max = np.mean(distances[:, -1])

    return r_min, r_max, s_min, s_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input point cloud")
    parser.add_argument(
        "--n-steps", default=10, type=int, help="Number of steps"
    )

    args = parser.parse_args()
    X = load(args.INPUT)

    assert X is not None, RuntimeError(
        f"Unable to handle input file {args.INPUT}"
    )

    # TODO: make configurable
    # - seed
    # - number of query points
    # - number of samples of data set
    rng = np.random.default_rng(42)
    X = X[rng.choice(X.shape[0], 10000, replace=False)]
    query_points = X[rng.choice(X.shape[0], 100, replace=False)]

    r_min, r_max, s_min, s_max = estimate_scales(X)
    # print(r_min, r_max, s_min, s_max)

    # TODO: Make configurable
    dim = 2
    n_steps = args.n_steps

    euclidicity = Euclidicity(
        r_min,
        r_max,
        s_min,
        s_max,
        max_dim=dim,
        n_steps=n_steps,
        method="ripser",
        X=X,
    )

    def _process(x):
        values = euclidicity(X, x)
        score = np.mean(np.nan_to_num(values))

        s = " ".join(str(a) for a in x)
        s += f" {score}"

        # TODO: Could also do something smarter here
        print(s)
        return score

    scores = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_process)(x) for x in query_points
    )
