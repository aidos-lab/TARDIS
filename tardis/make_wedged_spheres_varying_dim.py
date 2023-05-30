

"""Create "wedged spheres of possibly different dimensions" data set.
Usage:
    python make_wedged_spheres_varying_dim.py > Wedged_spheres_varying_dim.csv
"""

import argparse
import sys

import numpy as np

from toast.shapes import sample_from_wedged_sphere_varying_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d1", "--dimension1", default=1, type=int, help="Intrinsic dimension of first sphere"
    )
    parser.add_argument(
        "-d2", "--dimension2", default=2, type=int, help="Intrinsic dimension of second sphere"
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=10000,
        type=int,
        help="Number of samples",
    )

    args = parser.parse_args()

    X = sample_from_wedged_sphere_varying_dim(args.num_samples, args.dimension1, args.dimension2)
    np.savetxt(sys.stdout, X)
