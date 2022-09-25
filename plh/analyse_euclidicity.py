"""Basic statistical analysis of Euclidicity scores."""

import argparse

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def detect_outliers(data):
    """Detect outliers based on IQR criterion."""
    # Simple outlier detection: clip everything that is larger than
    # q3 + 1.5 * IQR.
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    upper = (data > q3 + 1.5 * iqr)
    lower = (data < q1 - 1.5 * iqr)

    print("- Found", upper.sum(), "upper outliers")
    print("- Found", lower.sum(), "lower outliers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", nargs="+", help="Input filename(s)")

    args = parser.parse_args()

    n_files = len(args.FILE)
    fig, axes = plt.subplots(nrows=2, ncols=n_files, squeeze=False)

    for (
        col,
        filename,
    ) in enumerate(args.FILE):
        X = np.loadtxt(filename)
        euclidicity = X[:, -1].flatten()

        detect_outliers(euclidicity)

        sns.histplot(data=euclidicity, kde=True, ax=axes[0, col])
        sns.violinplot(data=euclidicity, ax=axes[1, col])
        sns.stripplot(data=euclidicity, ax=axes[1, col], color="black", size=1)

    plt.show()
