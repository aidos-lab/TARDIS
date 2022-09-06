"""Wrappers for persistent homology calculations.

The purpose of this module is to provide wrappers for the persistent
homology calculations. This is to ensure that the returned shapes of
barcodes etc. are always consistent regardless of any implementation
details.
"""

import gudhi as gd
import numpy as np

from gph import ripser_parallel


class GUDHI:
    """Wrapper for GUDHI persistent homology calculations."""

    def __call__(self, X, max_dim):
        """Calculate persistent homology.

        Parameters
        ----------
        X : np.array of shape ``(N, d)``
            Input data set.

        max_dim : int
            Maximum dimension for calculations

        Returns
        -------
        np.array
            Full barcode (persistence diagram) of the data set.
        """
        barcodes = (
            gd.RipsComplex(points=X)
            .create_simplex_tree(max_dimension=max_dim)
            .persistence()
        )

        if len(barcodes) == 0:
            return None, -1

        # TODO: Check whether this is *always* a feature of non-zero
        # persistence.
        max_dim = np.max([d for d, _ in barcodes])

        # TODO: We are throwing away dimensionality information; it is
        # thus possible that we are matching across different dimensions
        # in any distance calculation.
        barcodes = np.asarray([np.array(x) for _, x in barcodes])

        return barcodes, max_dim

    def distance(self, D1, D2):
        """Calculate Bottleneck distance between two persistence diagrams."""
        return gd.bottleneck_distance(D1, D2)


class Ripser:
    def __call__(self, X, max_dim):
        if len(X) == 0:
            return [], -1

        diagrams = ripser_parallel(
            X, maxdim=max_dim, collapse_edges=True, n_threads=-1
        )

        diagrams = diagrams["dgms"]

        max_dim = np.max([d for d, D in enumerate(diagrams) if len(D) > 0])

        diagrams = np.row_stack(diagrams)

        return diagrams, max_dim

    def distance(self, D1, D2):
        return gd.bottleneck_distance(D1, D2)
