import numpy as np
import sys

from math import pi
from math import cos
from math import sin

from plh.euclidicity import Euclidicity

from tqdm import tqdm

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

rng = np.random.default_rng(42)

X = X[rng.choice(X.shape[0], 10000, replace=False)]
query_points = X[rng.choice(X.shape[0], 1000, replace=False)]

euclidicity = Euclidicity(
    0.05, 0.45, 0.2, 0.6, 2, n_steps=10, method="ripser", X=X
)

for x in tqdm(query_points, desc="Point"):
    values = euclidicity(X, x)
    score = np.mean(np.nan_to_num(values))

    s = " ".join(str(a) for a in x)
    s += f" {score}"
    tqdm.write(s)

    sys.stdout.flush()
