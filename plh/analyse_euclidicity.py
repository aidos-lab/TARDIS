"""Basic statistical analysis of Euclidicity scores.

This is a helper script for analysing Euclidicity scores. It generates
plots of the summary statistics and performs Tukey's range test.
"""

import argparse

import numpy as np

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

    print("IQR", iqr)

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
            X = np.loadtxt(filename, delimiter=",", skiprows=1)
        else:
            X = np.loadtxt(filename)

        euclidicity = X[:, -1].flatten()

        distributions.append(np.asarray(euclidicity))

        detect_outliers(euclidicity)
        print_summary_statistics(euclidicity)

        sns.histplot(data=euclidicity, kde=True, ax=axes[0, col])
        sns.violinplot(data=euclidicity, ax=axes[1, col])
        sns.stripplot(data=euclidicity, ax=axes[1, col], color="black", size=1)

    print(tukey_hsd(*distributions))
    plt.show()
