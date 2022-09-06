"""Shape sampling methods."""

import numpy as np


def sample_from_annulus(n, r, R, seed=None):
    """Sample points from a 2D annulus.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.

    Parameters
    ----------
    n : int
        Number of points to sample

    r : float
        Inner radius of annulus

    R : float
        Outer radius of annulus

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    np.array of shape `(n, 2)`
        Array containing sampled coordinates.
    """
    if r >= R:
        raise RuntimeError(
            "Inner radius must be less than or equal to outer radius"
        )

    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, n)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(rng.uniform(r**2, R**2, n))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X
