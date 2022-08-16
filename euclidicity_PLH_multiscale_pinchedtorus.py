from math import cos, sin, fmod, pi, sqrt

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import scipy.stats as ss

# create samples from a pinched torus in a pandas dataframe df, then conduct the same experiment as
# in 'euclidicity_PLH_multiscale_2Dsphere.py', but now for the pinched torus.
# Observation: bottleneck distances get larger when approaching the singularity.
# --> This approach seems to be more appropriate to detect singularities.

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

data = df.iloc[np.random.randint(0, len(df), 32000), :]
data = torch.tensor(data.values)

l1 = []
l2 = []

idx = np.random.randint(0, len(data), 1000)

for point in data[idx]:
    bd = []
    for r in np.arange(.05, .45, .05):
        for s in np.arange(.2, .6, .05):
            if r < s:
                locus = np.array(
                    [np.array(d) for d in data if np.linalg.norm(point - d) <= s and np.linalg.norm(point - d) >= r])
                barcodes = gd.RipsComplex(points=locus).create_simplex_tree(max_dimension=2).persistence()
                if len(barcodes) > 0:
                    barcodes = np.array(barcodes)
                    barcodes = np.array([np.array(x) for x in barcodes[:, 1]])

                euc_locus = torch_topological.data.sample_from_annulus(len(locus), r, s)
                barcodes_euc = gd.RipsComplex(points=euc_locus).create_simplex_tree(max_dimension=2).persistence()
                if len(barcodes_euc) > 0:
                    barcodes_euc = np.array(barcodes_euc)
                    barcodes_euc = np.array([np.array(x) for x in barcodes_euc[:, 1]])
                if len(barcodes) > 0 and len(barcodes_euc) > 0:
                    bd.append(gd.bottleneck_distance(barcodes, barcodes_euc))
    if len(bd) > 0:
        l2.append(sum(bd) / len(bd))
        l1.append(np.array(point))

l1 = np.array(l1)
l2 = np.array(l2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

points = ax.scatter(l1[:, 0], l1[:, 1], l1[:, 2], c=l2, cmap="plasma")

fig.colorbar(points)
plt.show()