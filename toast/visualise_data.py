"""Basic visualisation of Euclidicity.

This is a helper script for visualising Euclidicity scores of
high-dimensional point clouds.
"""

import argparse

import numpy as np

import phate

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", nargs="+", help="Input filename(s)")

    args = parser.parse_args()

    n_files = len(args.FILE)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(ncols=n_files)

    if n_files == 1:
        axes = [axes]

    # Following the parameters of the original PHATE publication. We set
    # a random state to ensure that the output remains reproducible.
    emb = phate.PHATE(decay=10, t=100, random_state=42)

    for filename, ax in zip(args.FILE, axes):
        X = np.loadtxt(filename)
        y = X[:, -1].flatten()

        # Remove Euclidicity scores. Our implementation adds them to the
        # last column of the data.
        X = X[:, :-1]

        X_emb = emb.fit_transform(X)

        scatter = ax.scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=y, s=5.0)
        fig.colorbar(scatter, ax=ax)

    plt.show()
