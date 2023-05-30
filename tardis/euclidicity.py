"""Euclidicity example implementation."""

import numpy as np

from sklearn.neighbors import KDTree

from toast.persistent_homology import GUDHI
from toast.persistent_homology import Ripser


class Euclidicity:
    """Functor for calculating Euclidicity of a point cloud."""

    def __init__(
        self,
        max_dim,
        r=None,
        R=None,
        s=None,
        S=None,
        n_steps=10,
        data=None,
        method="gudhi",
        model_sample_fn=None,
    ):
        """Initialise new instance of functor.

        Sets up a new instance of the Euclidicity functor and stores
        shared parameters that will be used for the calculation. The
        client has the choice of either providing global parameters,
        or adjusting them on a per-point basis.

        Parameters
        ----------
        max_dim : int
            Maximum dimension for persistent homology approximations.
            This is the *only* required parameter.

        r : float, optional
            Minimum inner radius of annulus

        R : float, optional
            Maximum inner radius of annulus

        s : float, optional
            Minimum outer radius of annulus

        S : float, optional
            Maximum outer radius of annulus

        n_steps : int, optional
            Number of steps for the radius parameter grid of the
            annulus. Note that the complexity of the function is
            quadratic in the number of steps.

        data : np.array or None
            If set, prepares a tree for nearest-neighbour and radius
            queries on the input data set. This can lead to substantial
            speed improvements in practice.

        method : str
            Persistent homology calculation method. At the moment, only
            "gudhi" and "ripser" are supported. "gudhi" is better for a
            small, low-dimensional data set, while "ripser" scales well
            to larger, high-dimensional point clouds.

        model_sample_fn : callable
            Function to be called for sampling from a comparison space.
            The function is being supplied with the number of samples,
            the radii of the annulus, and the intrinsic dimension. Its
            output must be a point cloud representing the annulus. If no
            sample function is provided, the class will default to
            compare the topological features with those of fixed
            Euclidean annulus.
        """
        self.r = r
        self.R = R
        self.s = s
        self.S = S

        self.n_steps = n_steps
        self.max_dim = max_dim

        self.model_sample_fn = model_sample_fn

        if method == "gudhi":
            self.vr = GUDHI()
        elif method == "ripser":
            self.vr = Ripser()
        else:
            raise RuntimeError("No persistent homology calculation selected.")

        # Prepare KD tree to speed up annulus calculations. We make this
        # configurable to permit both types of workflows.
        if data is not None:
            self.tree = KDTree(data)
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

        Other Parameters
        ----------------
        r : float, optional
            Minimum inner radius of annulus. Will default to global `r`
            parameter if not set.

        R : float, optional
            Maximum inner radius of annulus. Will default to global `R`
            parameter if not set.

        s : float, optional
            Minimum outer radius of annulus. Will default to global `s`
            parameter if not set.

        S : float, optional
            Maximum outer radius of annulus. Will default to global `S`
            parameter if not set.

        Returns
        -------
        Tuple of np.array, np.array
            1D array containing Euclidicity estimates. The length of the
            array depends on the number of scales. The second array will
            contain the persistent intrinsic dimension (PID) values.
        """
        r = kwargs.get("r", self.r)
        R = kwargs.get("R", self.R)
        s = kwargs.get("s", self.s)
        S = kwargs.get("S", self.S)

        bottleneck_distances = []
        dimensions = []

        for r in np.linspace(r, R, self.n_steps):
            for s in np.linspace(s, S, self.n_steps):
                if r < s:
                    dist, dim = self._calculate_euclidicity(
                        r, s, X, x, self.max_dim
                    )

                    bottleneck_distances.append(dist)
                    dimensions.append(dim)

        return np.asarray(bottleneck_distances), np.asarray(dimensions)

    # Auxiliary method for performing the 'heavy lifting' when it comes
    # to Euclidicity calculations.
    def _calculate_euclidicity(self, r, s, X, x, d):
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

        barcodes, max_dim = self.vr(annulus, d)

        if max_dim < 0:
            return np.nan, max_dim

        if self.model_sample_fn is not None:
            euclidean_annulus = self.model_sample_fn(
                n=len(annulus), r=r, R=s, d=d
            )
            barcodes_euclidean, _ = self.vr(euclidean_annulus, d)

        # No sampling function has been specified. Compare to a fixed
        # annulus with known persistent homology.
        #
        # TODO: Technically, the single feature should be put into
        # a persistence diagram of the right dimension. Let us not
        # do that for now (since we stack diagrams anyway).
        else:
            barcodes_euclidean = np.asarray([[0, np.inf], [r, s]])

        if barcodes_euclidean is None:
            return np.nan, max_dim

        dist = self.vr.distance(barcodes, barcodes_euclidean)
        return dist, max_dim
