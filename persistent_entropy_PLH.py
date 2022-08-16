# specify inner radius r, and outer radius s.
r = .3
s = .5

# sample from concatenated spheres
data = sample_from_concat_sphere(n=2000, d=1)

l1 = []
l2 = []

# set index to randomly generate a smaller subsample
idx = np.random.randint(0, len(data), 500)

for point in data[idx]:
    # calculate the intrinsic locus of the data with point as center, and w.r.t. the radii r and s.
    locus = np.array([np.array(d) for d in data if np.linalg.norm(point - d) <= s and np.linalg.norm(point - d) >= r])
    # calculate barcodes of the locus
    barcodes = gd.RipsComplex(points=locus).create_simplex_tree(max_dimension=2).persistence()
    # list of points under consideration
    l1.append(np.array(point))
    # list of persistent_entropy in dim 0 of the respective points
    l2.append(persistent_entropy(barcodes, dim=0))

l1 = np.array(l1)
l2 = np.array(l2)

# creates plot that shows the subsamples with the corresponding persistent entropy as heat.
f, ax = plt.subplots()

points = ax.scatter(x=l1[:,0], y=l1[:,1], c=l2, cmap="plasma")
f.colorbar(points)

plt.show()

# Remark: Setting r = .05 and s = .5 e.g. leads to different results that seem to detect the singular point better.
# Idea: Consider PLH for varying radii r and s. --> see 'persistent_entropy_PLH_multiscale.py'