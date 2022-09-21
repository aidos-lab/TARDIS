"""Command-line interface for Euclidicity calculations.

This script is the main command-line interface for our Euclidicity
calculations. It supports loading various input formats, for which
it will calculate Euclidicity scores.
"""

import argparse
import colorlog
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


def setup():
    """Perform logging and argument parsing setup.

    Sets up the command-line interface for subsequent usage so that we
    do not clutter up the actual Euclidicity calculations.

    Returns
    -------
    Tuple of logger and parsed arguments
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)-.1s: %(message)s")
    )

    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(colorlog.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input point cloud")

    euclidicity_group = parser.add_argument_group("Euclidicity calculations")

    euclidicity_group.add_argument(
        "-d",
        "--dimension",
        default=2,
        type=int,
        help="Known or estimated intrinsic dimension",
    )
    euclidicity_group.add_argument(
        "-r",
        type=float,
        help="Minimum inner radius of annulus",
    )
    euclidicity_group.add_argument(
        "-R",
        type=float,
        help="Maximum inner radius of annulus",
    )
    euclidicity_group.add_argument(
        "-s",
        type=float,
        help="Minimum outer radius of annulus",
    )
    euclidicity_group.add_argument(
        "-S",
        type=float,
        help="Maximum outer radius of annulus",
    )
    euclidicity_group.add_argument(
        "--n-steps",
        default=10,
        type=int,
        help="Number of steps for annulus sampling",
    )

    sampling_group = parser.add_argument_group("Sampling")

    sampling_group.add_argument(
        "-b",
        "--batch-size",
        default=10000,
        type=int,
        help="Number of points to sample from input data",
    )
    sampling_group.add_argument(
        "-q",
        "--num-query-points",
        default=1000,
        type=int,
        help="Number of query points for Euclidicity calculations",
    )
    sampling_group.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for reproducible experiments",
    )

    args = parser.parse_args()
    return logger, args


if __name__ == "__main__":
    logger, args = setup()

    X = load(args.INPUT)

    assert X is not None, RuntimeError(
        f"Unable to handle input file {args.INPUT}"
    )

    if args.seed is not None:
        logger.info(f"Using pre-defined seed {args.seed}")

    rng = np.random.default_rng(args.seed)

    logger.info(f"Sampling a batch of {args.batch_size} points")
    logger.info(f"Using {args.num_query_points} query points")

    X = X[rng.choice(X.shape[0], args.batch_size, replace=False)]
    query_points = X[
        rng.choice(X.shape[0], args.num_query_points, replace=False)
    ]

    r, R, s, S = args.r, args.R, args.s, args.S

    # Check whether we have to perform scale estimation on a per-point
    # basis. If not, we just supply an empty dict.
    if all([x is not None for x in [r, R, s, S]]):
        logger.info(
            f"Using global scales r = {r:.2f}, R = {R:.2f}, "
            f"s = {s:.2f}, S = {S:.2f}"
        )

        scales = [dict()] * len(query_points)
    else:
        logger.info(
            "Performing scale estimation since no parameters "
            "have been provided by the client."
        )

        scales = estimate_scales(X, query_points, 50)

    max_dim = args.dimension
    n_steps = args.n_steps

    logger.info(f"Maximum dimension: {max_dim}")
    logger.info(f"Number of steps for local sampling: {n_steps}")

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
