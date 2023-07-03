"""Simple example of integrating TARDIS and ``numpy``."""


import numpy as np

from tardis import calculate_euclidicity


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # This is the same data set that will also be used for the
    # estimation of Euclidicity later on.
    X = rng.normal(size=(100, 3))

    # Only get Euclidicity values. By default, no dimensions will be
    # returned (they are always computed, though). Use `n_steps` for
    # controlling the scale traversal.
    euclidicity = calculate_euclidicity(
        X, r=0.01, R=0.25, s=0.05, S=0.5, max_dim=3, n_steps=5
    )

    # Get both Euclidicity and the persistent intrinsic dimension (PID)
    # of each data point.
    euclidicity, persistent_intrinsic_dimension = calculate_euclidicity(
        X,
        r=0.01,
        R=0.25,
        s=0.05,
        S=0.5,
        max_dim=3,
        n_steps=5,
        return_dimensions=True,
    )

    # Let's calculate Euclidicity with respect to *another* data set.
    Y = rng.normal(size=(10, 3))
    euclidicity = calculate_euclidicity(
        X, Y, r=0.01, R=0.25, s=0.05, S=0.5, max_dim=3
    )
