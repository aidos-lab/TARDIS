"""Create "pinched torus" data set.

Usage:
    python make_pinched_torus.py > Pinched_torus.csv
"""

import sys

from math import pi
from math import cos
from math import sin

import numpy as np

n = 4096
m = 512
o = 512
R = 10
r = 1
k = 0.5

# Gap size in angular coordinates. This is to be seen as the radius for
# which the 'pinch' is relevant.
gap_size = pi / 180.0 * 90

X = list()
Y = list()
Z = list()

for i in range(m):
    for j in range(o):
        phi = 2 * pi * i / (m - 1)
        theta = 2 * pi * j / (o - 1)

        r_ = r

        x = (R + r_ * cos(theta) * cos(k * phi)) * cos(phi)
        y = (R + r_ * cos(theta) * cos(k * phi)) * sin(phi)
        z = r_ * sin(theta) * cos(k * phi)

        X.append(x)
        Y.append(y)
        Z.append(z)

X = np.vstack((X, Y, Z)).T
np.savetxt(sys.stdout, X)
