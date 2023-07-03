"""Main entry point for API calls to TARDIS.

This module collects API calls to TARDIS. Each exported function should
facilitate using TARDIS for data analysis. Users that need fine-grained
control are encouraged to build their own functions.
"""

import joblib

import numpy as np

from tardis.euclidicity import Euclidicity
from tardis.utils import estimate_scales


def calculate_euclidicity(
    X,
    Y=None,
    max_dim=2,
    n_steps=10,
    r=None,
    R=None,
    s=None,
    S=None,
    k=20,
    n_jobs=1,
    return_dimensions=False,
):
    """Convenience function for calculating Euclidicity of a point cloud.

    This function provides the most convenient interface for calculating
    Euclidicity of a point cloud. Internally, this function will use the
    best and fastest Euclidicity calculation, but this comes at the cost
    of configurability.

    TODO: Document me :-)
    """
    r_, R_, s_, S_ = r, R, s, S
    query_points = X if Y is None else Y

    if all([x is not None for x in [r_, R_, s_, S_]]):
        scales = [dict()] * len(query_points)
    else:
        scales = estimate_scales(X, query_points, k)

    euclidicity = Euclidicity(
        max_dim=max_dim,
        n_steps=n_steps,
        r=r_,
        R=R_,
        s=s_,
        S=S_,
        method="ripser",
        data=X,
    )

    def _process(x, scale=None):
        scores, dimensions = euclidicity(X, x, **scale)

        score = np.mean(np.nan_to_num(scores))
        dimension = np.mean(dimensions)

        return score, dimension

    output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_process)(x, scale)
        for x, scale in zip(query_points, scales)
    )

    euclidicity = np.asarray([e for (e, _) in output])
    persistent_intrinsic_dimension = np.asarray([d for (_, d) in output])

    if return_dimensions:
        return euclidicity, persistent_intrinsic_dimension
    else:
        return euclidicity
