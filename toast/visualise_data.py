"""Basic visualisation of Euclidicity.

This is a helper script for visualising Euclidicity scores of
high-dimensional point clouds.
"""

import argparse
import os

import numpy as np
import pandas as pd

import phate

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("FILE", nargs="+", type=str, help="Input filename(s)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory. If set, will store embedded point clouds.",
        type=str,
    )

    args = parser.parse_args()

    n_files = len(args.FILE)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(ncols=n_files)

    if n_files == 1:
        axes = [axes]

    # Following the parameters of the original PHATE publication. We set
    # a random state to ensure that the output remains reproducible.
    emb = phate.PHATE(decay=10, t=50, random_state=42)

    for filename, ax in zip(args.FILE, axes):
        if os.path.splitext(filename)[1] == ".csv":
            df = pd.read_csv(filename)
            df = df.drop("persistent_intrinsic_dimension", axis="columns")
            X = df.to_numpy()
        else:
            X = np.loadtxt(filename)

        y = X[:, -1].flatten()

        iqr = np.subtract(*np.percentile(y, [75, 25]))
        q3 = np.percentile(y, 75)

        # Remove Euclidicity scores. Our implementation adds them to the
        # last column of the data.
        X = X[:, :-1]

        X_emb = emb.fit_transform(X)

        scatter = ax.scatter(
            x=X_emb[:, 0],
            y=X_emb[:, 1],
            c=y,
            alpha=0.5,
            s=1.0,
            # Try to highlight outliers a little bit better.
            vmax=q3 + 1.5 * iqr,
        )
        fig.colorbar(scatter, ax=ax)

        if args.output is not None:
            out_filename = os.path.basename(filename)
            out_filename = os.path.splitext(out_filename)[0] + ".csv"
            out_filename = os.path.join(args.output, out_filename)

            X_out = np.hstack((X_emb, y.reshape(-1, 1)))

            np.savetxt(
                out_filename,
                X_out,
                fmt="%.4f",
                delimiter=",",
                header="x,y,euclidicity",
            )

    plt.show()
