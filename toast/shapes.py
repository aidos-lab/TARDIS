"""Shape sampling methods."""

import numpy as np


def sample_from_annulus(n, r, R, d=2, seed=None):
    """Sample points from an annulus.

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

    d : int
        Dimension of the annulus. Technically, for higher dimensions, we
        should call the resulting space a "hyperspherical shell." Notice
        that the algorithm for sampling points in higher dimensions uses
        rejection sampling, so its efficiency decreases as the dimension
        increases.

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

    if d == 2:
        thetas = rng.uniform(0, 2 * np.pi, n)

        # Need to sample based on squared radii to account for density
        # differences.
        radii = np.sqrt(rng.uniform(r**2, R**2, n))

        X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    else:
        X = np.empty((1, d))

        while True:
            sample = sample_from_ball(n, d, r=R, seed=seed)
            norms = np.sqrt(np.sum(np.abs(sample) ** 2, axis=-1))

            X = np.row_stack((X, sample[norms >= r]))

            if len(X) >= n:
                X = X[:n]
                break

    return X


def sample_from_ball(n=100, d=2, r=1, seed=None):
    """Sample `n` data points from a `d`-ball in `d` dimensions.

    Parameters
    -----------
    n : int
        Number of data points in ball.

    d : int
        Dimension of the ball. Notice that there is an inherent shift in
        dimension if you compare a ball to a sphere.

    r : float
        Radius of ball.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    np.array of shape `(n, d)`
        Array of sampled coordinates.

    References
    ----------
    .. [Voelker2007] A. Voelker et al, Efficiently sampling vectors and
    coordinates from the $n$-sphere and $n$-ball, Technical Report,
    2017. http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    rng = np.random.default_rng(seed)

    # This algorithm was originally described in the following blog
    # post:
    #
    # http://extremelearning.com.au/how-to-generate-uniformly-random-points
    # -on-n-spheres-and-n-balls/
    #
    # It's mind-boggling that this works but it's true!
    U = rng.normal(size=(n, d + 2))
    norms = np.sqrt(np.sum(np.abs(U) ** 2, axis=-1))
    U = r * U / norms[:, np.newaxis]
    X = U[:, 0:d]

    return np.asarray(X)


def sample_from_sphere(n=100, d=2, r=1, noise=None, seed=None):
    """Sample `n` data points from a `d`-sphere in `d + 1` dimensions.

    Parameters
    -----------
    n : int
        Number of data points in shape.

    d : int
        Dimension of the sphere.

    r : float
        Radius of sphere.

    noise : float or None
        Optional noise factor. If set, data coordinates will be
        perturbed by a standard normal distribution, scaled by
        `noise`.

    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.

    Returns
    -------
    np.array of shape `(n, d + 1)`
        Array of sampled coordinates.

    Notes
    -----
    This function was originally authored by Nathaniel Saul as part of
    the `tadasets` package. [tadasets]_

    References
    ----------
    .. [tadasets] https://github.com/scikit-tda/tadasets
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d + 1))

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * rng.standard_normal(data.shape)

    return np.asarray(data)


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


def sample_from_wedged_sphere_varying_dim(n=100, d1=1, d2=2, r=1, noise=None):
    """Sample points from two wedged spheres of possibly different dimensions.

    This function permits sampling from wedged spheres of different
    dimensions, thus making it possible to, for instance, combine a
    circle with an ordinary 2D sphere.

    Parameters
    ----------
    n : int
        Number of points to sample

    d1 : int
        Intrinsic dimension of first sphere. The ambient dimension will be
        ``d1 + 1``.

    d2 : int
        Intrinsic dimension of second spheres. The ambient dimension will be
        ``d2 + 1``.

    r : float
        Radius of spheres

    noise : float or None
        If set, will be used as a scale factor for random perturbations
        of the positions of points, following a standard normal
        distribution.
    """
    data1 = np.random.randn(n, d1 + 1)
    data1 = r * data1 / np.sqrt(np.sum(data1**2, 1)[:, None])
    zeros = np.zeros((len(data1), d2 - d1))
    data1 = np.concatenate((data1, zeros), axis=1)

    data2 = np.random.randn(n, d2 + 1)
    data2 = (
        r * data2 / np.sqrt(np.sum(data2**2, 1)[:, None])
    ) + np.concatenate((np.array([2 * r]), np.zeros(data2.shape[1] - 1)))

    data = np.concatenate((data1, data2))
    if noise:
        data += noise * np.random.randn(*data.shape)

    return data


sample_from_annulus(100, 0.1, 0.2, d=3, seed=42)
