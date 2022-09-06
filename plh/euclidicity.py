"""Euclidicity example implementation."""

import gudhi as gd
import numpy as np

from plh.persistent_homology import GUDHI
from plh.shapes import sample_from_annulus


class Euclidicity:
    """Functor for calculating Euclidicity of a point cloud."""

    def __init__(self, r, R, s, S, max_dim, n_steps=100):
        """Initialise new instance of functor.

        Sets up a new instance of the Euclidicity functor and stores
        shared parameters that will be used for the calculation.

        Parameters
        ----------
        r : float
            Minimum inner radius of annulus

        R : float
            Maximum inner radius of annulus

        s : float
            Minimum outer radius of annulus

        S : float
            Maximum outer radius of annulus

        max_dim : int
            Maximum dimension for persistent homology approximations.

        n_steps : int
            Number of steps for the radius parameter grid of the
            annulus. Note that the complexity of the function is
            quadratic in the number of steps.
        """
        self.r = r
        self.R = R
        self.s = s
        self.S = S

        # TODO: raise some nice and informative `RuntimeError` instances
        # here. Could potentially also try to constrain these values.
        assert r >= 0
        assert s >= 0
        assert r <= R
        assert s <= S
        assert R <= S

        self.n_steps = n_steps
        self.max_dim = max_dim

        # TODO: Make this configurable.
        self.vr = GUDHI()

    def __call__(self, X, x):
        """Calculate Euclidicity of a specific point.

        Parameters
        ----------
        X : np.array or tensor of shape ``(N, d)``
            Input data set. Must be compatible with the persistent
            homology calculations.

        x : np.array, tensor, or iterable of shape ``(d, )``
            Input point.

        Returns
        -------
        np.array
            Array containing Euclidicity estimates.
        """
        bottleneck_distances = []
        for r in np.linspace(self.r, self.R, self.n_steps):
            for s in np.linspace(self.s, self.S, self.n_steps):
                if r < s:
                    dist, _ = self._calculate_euclidicity(r, s, X, x)
                    bottleneck_distances.append(dist)

        return np.asarray(bottleneck_distances)

    # Auxiliary method for performing the 'heavy lifting' when it comes
    # to Euclidicity calculations.
    def _calculate_euclidicity(self, r, s, X, x):
        # TODO: Turn this into a KD tree query; that way, we can benefit
        # from pre-calculated things.
        annulus = np.asarray(
            [
                np.asarray(p)
                for p in X
                if np.linalg.norm(x - p) <= s and np.linalg.norm(x - p) >= r
            ]
        )

        # TODO: This can be offloaded to a general persistent homology
        # calculation method. We just need to make sure that the output
        # is always provided in the same way.
        barcodes = (
            gd.RipsComplex(points=annulus)
            .create_simplex_tree(max_dimension=self.max_dim)
            .persistence()
        )

        barcodes, max_dim = self.vr(annulus, self.max_dim)

        if max_dim < 0:
            return np.nan, max_dim

        euclidean_annulus = sample_from_annulus(len(annulus), r, s)
        barcodes_euclidean, _ = self.vr(euclidean_annulus, self.max_dim)

        if barcodes_euclidean is None:
            return np.nan, max_dim

        dist = self.vr.distance(barcodes, barcodes_euclidean)
        return dist, max_dim


########################################################################
# HIC SVNT DRACONES
########################################################################

rng = np.random.default_rng(42)
X = rng.uniform(low=0.0, high=1.0, size=(100, 2))
x = (0.5, 0.5)

euclidicity = Euclidicity(0.05, 0.25, 0.1, 0.5, 2, n_steps=10)
values = euclidicity(X, x)
print(values)
