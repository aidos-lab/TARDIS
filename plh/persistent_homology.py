"""Wrappers for persistent homology calculations.

The purpose of this module is to provide wrappers for the persistent
homology calculations. This is to ensure that the returned shapes of
barcodes etc. are always consistent regardless of any implementation
details.
"""

import gudhi as gd
import numpy as np


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

        barcodes = np.asarray([np.array(x) for _, x in barcodes])

        if len(barcodes) > 0:
            # TODO: Check whether this is *always* a feature of non-zero
            # persistence.
            max_dim = np.max([d for d, _ in barcodes])

            return barcodes, max_dim

        return None, -1
