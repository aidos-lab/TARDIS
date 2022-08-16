# sample data from concatenated 2-spheres and calculate multiscale (for varyiing radii) bottleneck
# distances between the neighbourhood of a point and a euclidean model space annulus.
# Observation: bottleneck distances get larger when approaching the singularity.

data = sample_from_concat_sphere(n=4000, d=2)

l1 = []
l2 = []

idx = np.random.randint(0, len(data), 1000)

for point in data[idx]:
    bd = []
    for r in np.arange(.05, .35, .05):
        for s in np.arange(.1, .5, .05):
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
    l2.append(sum(bd) / len(bd))
    l1.append(np.array(point))

l1 = np.array(l1)
l2 = np.array(l2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

points = ax.scatter(l1[:, 0], l1[:, 1], l1[:, 2], c=l2, cmap="plasma")

fig.colorbar(points)
plt.show()