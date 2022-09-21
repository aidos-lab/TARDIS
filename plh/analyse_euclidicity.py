"""Basic statistical analysis of Euclidicity scores."""

import argparse

import numpy as np

from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", nargs="+", help="Input filename(s)")

    args = parser.parse_args()

    n_files = len(args.FILE)
    fig, axes = plt.subplots(ncols=n_files)

    if n_files == 1:
        axes = [axes]

    emb = MDS()

    for filename, ax in zip(args.FILE, axes):
        X = np.loadtxt(filename)
        euclidicity = X[:, -1].flatten()

        X_emb = emb.fit_transform(X)
        sns.scatterplot(x=X_emb[:, 0],  y=X_emb[:, 1], hue=euclidicity)

    plt.show()
