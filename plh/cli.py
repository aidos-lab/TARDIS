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


def estimate_scales(X, query_points, k_max):
    """Perform simple scale estimation of the data set.

    Parameters
    ----------
    k_max : int
        Maximum number of neighbours to consider for the local scale
        estimation.

    Returns
    --------
    List of dict
        A list of dictionaries consisting of the minimum and maximum
        inner and outer radius, respectively.
    """
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    distances, _ = tree.query(query_points, k=k_max, return_distance=True)

    # Ignore the distance to ourself, as we know that one already.
    distances = distances[:, 1:]

    # TODO: We could pick something smarter here...
    r_min = np.mean(distances[:, 0])
    r_max = np.mean(distances[:, 5])
    s_min = np.mean(distances[:, 5])
    s_max = np.mean(distances[:, -1])

    scales = [
        {
            "r": dist[0],
            "R": dist[round(k_max / 3)],
            "s": dist[round(k_max / 3)],
            "S": dist[-1],
        }
        for dist in distances
    ]

    return scales


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input point cloud")
    parser.add_argument(
        "--n-steps", default=10, type=int, help="Number of steps"
    )
    parser.add_argument(
        "-d",
        "--dimension",
        default=2,
        type=int,
        help="Intrinsic dimension (can be an estimate)",
    )
    parser.add_argument(
        "-r",
        type=float,
        help="Minimum inner radius of annulus",
    )
    parser.add_argument(
        "-R",
        type=float,
        help="Maximum inner radius of annulus",
    )
    parser.add_argument(
        "-s",
        type=float,
        help="Minimum outer radius of annulus",
    )
    parser.add_argument(
        "-S",
        type=float,
        help="Maximum outer radius of annulus",
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

    r, R, s, S = args.r, args.R, args.s, args.S

    # Check whether we have to perform scale estimation on a per-point
    # basis. If not, we just supply an empty dict.
    if all([x is not None for x in [r, R, s, S]]):
        scales = [dict()] * len(query_points)
    else:
        scales = estimate_scales(X, query_points, 50)

    max_dim = args.dimension
    n_steps = args.n_steps

    euclidicity = Euclidicity(
        max_dim=max_dim,
        n_steps=n_steps,
        r=args.r,
        R=args.R,
        s=args.s,
        S=args.S,
        method="ripser",
        data=X,
    )

    def _process(x, scale=None):
        values = euclidicity(X, x, **scale)
        score = np.mean(np.nan_to_num(values))

        s = " ".join(str(a) for a in x)
        s += f" {score}"

        # TODO: Could also do something smarter here
        print(s)
        return score

    scores = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_process)(x, scale)
        for x, scale in zip(query_points, scales)
    )
