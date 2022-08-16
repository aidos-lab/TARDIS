# same experiment as in 'persistent_entropy_PLH.py', but this time for varying
# parameters r and s.
# Observation: this approach leads to better results if one is interested in detecting the singular point

data = sample_from_concat_sphere(n=2000, d=1)

l1 = []
l2 = []

idx = np.random.randint(0, len(data), 500)

for point in data[idx]:
    pe = []
    for r in np.arange(.05, .35, .05):
        for s in np.arange(.1, .5, .05):
            if r < s:
                locus = np.array(
                    [np.array(d) for d in data if np.linalg.norm(point - d) <= s and np.linalg.norm(point - d) >= r])
                barcodes = gd.RipsComplex(points=locus).create_simplex_tree(max_dimension=2).persistence()
                pe.append(persistent_entropy(barcodes, dim=0))
    l2.append(sum(pe) / len(pe))
    l1.append(np.array(point))

l1 = np.array(l1)
l2 = np.array(l2)

f, ax = plt.subplots()

points = ax.scatter(x=l1[:, 0], y=l1[:, 1], c=l2, cmap="plasma")
f.colorbar(points)

plt.show()