from math import cos, sin, fmod, pi, sqrt

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import scipy.stats as ss

# create samples from a pinched torus in a pandas dataframe df, then conduct the same experiment as
# in 'persistent_entropy_PLH_multiscale.py', but now for the pinched torus.
# Observation: persistent entropy around the singularity is low for the pinched torus, whereas it is high
# for the concatenated spheres.
# Conclusion: persistent entropy is not able to detect singularities. --> need different approach.
# Idea: Use model spaces instead.
n = 4096
m = 512
o = 512
R = 10
r = 1
k = .5

# Gap size in angular coordinates. This is to be seen as the radius for
# which the 'pinch' is relevant.
gap_size = pi / 180.0 * 90

X = list()
Y = list()
Z = list()

for i in range(m):
    for j in range(o):
        phi   = 2*pi * i / (m-1)
        theta = 2*pi * j / (o-1)

        r_ = r

        x = (R+r_*cos(theta)*cos(k*phi))*cos(phi)
        y = (R+r_*cos(theta)*cos(k*phi))*sin(phi)
        z =  r_ * sin(theta) * cos(k*phi)

        X.append(x)
        Y.append(y)
        Z.append(z)

df = pd.DataFrame(
  {
   'x' : X,
   'y' : Y,
   'z' : Z
  }
)

data = df.iloc[np.random.randint(0, len(df), 3000), :]
data = torch.tensor(data.values)

l1 = []
l2 = []

idx = np.random.randint(0, len(data), 1000)

for point in data[idx]:
    pe = []
    for r in np.arange(.5, 5.5, 1):
        for s in np.arange(1, 7, 1):
            if r < s:
                locus = np.array(
                    [np.array(d) for d in data if np.linalg.norm(point - d) <= s and np.linalg.norm(point - d) >= r])
                barcodes = gd.RipsComplex(points=locus).create_simplex_tree(max_dimension=2).persistence()
                pe.append(persistent_entropy(barcodes, dim=1))
    l2.append(sum(pe) / len(pe))
    l1.append(np.array(point))

l1 = np.array(l1)
l2 = np.array(l2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

points = ax.scatter(l1[:, 0], l1[:, 1], l1[:, 2], c=l2, cmap="plasma")

fig.colorbar(points)
plt.show()