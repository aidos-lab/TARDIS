"""Euclidicity example implementation."""

import numpy as np

from sklearn.neighbors import KDTree

from plh.persistent_homology import GUDHI
from plh.persistent_homology import Ripser

from plh.shapes import sample_from_annulus


class Euclidicity:
    """Functor for calculating Euclidicity of a point cloud."""

    def __init__(
        self, r, R, s, S, max_dim, n_steps=100, method="gudhi", X=None
    ):
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

        method : str
            Persistent homology calculation method. TODO: Document me.

        X : np.array or None
            If set, prepares a tree for nearest-neighbour and radius
            queries on `X`, the input data set.
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

        if method == "gudhi":
            self.vr = GUDHI()
        elif method == "ripser":
            self.vr = Ripser()
        else:
            raise RuntimeError("No persistent homology calculation selected.")

        # Prepare KD tree to speed up annulus calculations. We make this
        # configurable to permit both types of workflows.
        if X is not None:
            self.tree = KDTree(X)
        else:
            self.tree = None

    def __call__(self, X, x, **kwargs):
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
        r = kwargs.get("r", self.r)
        R = kwargs.get("R", self.R)
        s = kwargs.get("s", self.s)
        S = kwargs.get("S", self.S)

        bottleneck_distances = []
        for r in np.linspace(r, R, self.n_steps):
            for s in np.linspace(s, S, self.n_steps):
                if r < s:
                    dist, _ = self._calculate_euclidicity(r, s, X, x)
                    bottleneck_distances.append(dist)

        return np.asarray(bottleneck_distances)

    # Auxiliary method for performing the 'heavy lifting' when it comes
    # to Euclidicity calculations.
    def _calculate_euclidicity(self, r, s, X, x):
        if self.tree is not None:
            inner_indices = self.tree.query_radius(x.reshape(1, -1), r)[0]
            outer_indices = self.tree.query_radius(x.reshape(1, -1), s)[0]

            annulus_indices = np.setdiff1d(outer_indices, inner_indices)
            annulus = X[annulus_indices]
        else:
            annulus = np.asarray(
                [
                    np.asarray(p)
                    for p in X
                    if np.linalg.norm(x - p) <= s
                    and np.linalg.norm(x - p) >= r
                ]
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
