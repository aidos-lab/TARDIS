"""Basic statistical analysis of Euclidicity scores."""

import argparse

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", nargs="+", help="Input filename(s)")

    args = parser.parse_args()

    n_files = len(args.FILE)
    fig, axes = plt.subplots(ncols=n_files)

    for filename, ax in zip(args.FILE, axes):
        X = np.loadtxt(filename)
        euclidicity = X[:, -1].flatten()

        sns.boxplot(euclidicity, ax=ax)

    plt.show()
