"""Basic statistical analysis of Euclidicity scores.

This is a helper script for analysing Euclidicity scores. It generates
plots of the summary statistics and performs Tukey's range test.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import tukey_hsd


def detect_outliers(data):
    """Detect outliers based on IQR criterion."""
    # Simple outlier detection: clip everything that is larger than
    # q3 + 1.5 * IQR.
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    print(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")

    upper = data > q3 + 1.5 * iqr
    lower = data < q1 - 1.5 * iqr

    print("- Found", upper.sum(), "upper outliers")
    print("- Found", lower.sum(), "lower outliers")


def print_summary_statistics(data):
    """Print some summary statistics."""
    print(
        f"max = {np.max(data):.2f}, "
        f"mean = {np.mean(data):.2f}, "
        f"min = {np.min(data):.2f}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", nargs="+", help="Input filename(s)")

    args = parser.parse_args()

    n_files = len(args.FILE)
    fig, axes = plt.subplots(nrows=2, ncols=n_files, squeeze=False)

    distributions = []

    for (
        col,
        filename,
    ) in enumerate(args.FILE):
        print(f"Processing {filename}")

        if filename.endswith(".csv"):
            df = pd.read_csv(filename)
            df = df.drop("persistent_intrinsic_dimension", axis="columns")
            X = df.to_numpy()
        elif filename.endswith(".npz"):
            X = np.load(filename)["arr_0"]
        else:
            X = np.loadtxt(filename)

        # Skip empty files because they lead to problems in the
        # downstream analysis.
        if len(X) == 0:
            continue

        euclidicity = X[:, -1].flatten()

        distributions.append(np.asarray(euclidicity))

        detect_outliers(euclidicity)
        print_summary_statistics(euclidicity)

        axes[0, col].set_title(os.path.splitext(os.path.basename(filename))[0])

        sns.histplot(data=euclidicity, kde=True, ax=axes[0, col])
        sns.violinplot(data=euclidicity, ax=axes[1, col])
        sns.stripplot(data=euclidicity, ax=axes[1, col], color="black", size=1)

    # We can only do this with more than one distribution, but even for
    # a single distribution, we can show the respective plot.
    if len(distributions) > 1:
        print(tukey_hsd(*distributions))

    plt.show()
