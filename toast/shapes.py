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


def sample_from_wedged_spheres(n=100, d=2, r=1, noise=None, seed=None):
    """Sample points from two wedged spheres.

    Parameters
    ----------
    n : int
        Number of points to sample

    d : int
        Intrinsic dimension of spheres. The ambient dimension will be
        ``d + 1``.

    r : float
        Radius of spheres

    noise : float or None
        If set, will be used as a scale factor for random perturbations
        of the positions of points, following a standard normal
        distribution.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.
    """
    rng = np.random.default_rng(seed)

    data1 = rng.standard_normal((n, d + 1))
    data1 = r * data1 / np.sqrt(np.sum(data1**2, 1)[:, None])

    data2 = rng.standard_normal((n, d + 1))
    data2 = (
        r * data2 / np.sqrt(np.sum(data2**2, 1)[:, None])
    ) + np.concatenate((np.array([2 * r]), np.zeros(data2.shape[1] - 1)))

    X = np.concatenate((data1, data2))

    if noise:
        X += noise * rng.standard_normal(X.shape)

    return X
