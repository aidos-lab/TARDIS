"""Convert .mat to .npy file(s).

Usage:
    python mat_to_npy.py INPUT
"""

import os

import numpy as np
import scipy.io as sio
import sys

fname = sys.argv[1]
X = sio.loadmat(fname)
X = X["data"]

fname = os.path.splitext(fname)[0]

if not os.path.exists(fname):
    np.savez(fname, data=X)
