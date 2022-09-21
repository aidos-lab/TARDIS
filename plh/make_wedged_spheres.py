"""Create "wedged spheres" data set.

Usage:
    python make_wedged_spheres.py 2 > Wedged_spheres_2.csv
"""

import argparse
import sys

import numpy as np

from plh.shapes import sample_from_wedged_spheres


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dimension", type=int, help="Intrinsic dimension"
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, help="Number of samples"
    )

    args = parser.parse_args()

    X = sample_from_wedged_spheres(args.num_samples, args.dimension)
    np.savetxt(sys.stdout, X)
