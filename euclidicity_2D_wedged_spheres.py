def sample_from_concat_sphere(n=100, d=2, r=1, noise=None):
    data1 = np.random.randn(n, d+1)
    data1 = r * data1 / np.sqrt(np.sum(data1**2, 1)[:, None])
    
    data2 = np.random.randn(n, d+1)
    data2 = (r * data2 / np.sqrt(np.sum(data2**2, 1)[:, None])) + np.concatenate((np.array([2*r]),np.zeros(data2.shape[1]-1)))
    
    data = np.concatenate((data1,data2))
    if noise:
        data += noise * np.random.randn(*data.shape)

    return torch.as_tensor(data)

# intrinsic dimension
d = 2

# number of points to compute euclidicity for
n = 1000

X = sample_from_concat_sphere(n=10000,d=d,r=1,noise=None)

idx = np.random.randint(0,len(X),n)

euclidicity_information_2 = []
x_coord = []
y_coord = []
z_coord = []
for x in X[idx]:
    euclidicity = Euclidicity(0.05, 0.25, 0.1, 0.5, d+2, n_steps=20, method="ripser")
    values = euclidicity(X, x)
    euclidicity_information_2.append(values)
    x_coord.append(x[0])
    y_coord.append(x[1])
    z_coord.append(x[2])
