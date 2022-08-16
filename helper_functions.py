import numpy as np
import torch
import matplotlib.pyplot as plt
import torch_topological.data
import gudhi as gd

# creates samples from two spheres glued together at a point; n: number of samples, d: dimension of spheres,
# r: radius of spheres, noise: adds random gaussian noise to the samples if set to true.
def sample_from_concat_sphere(n=100, d=2, r=1, noise=None):
    data1 = np.random.randn(n, d + 1)
    data1 = r * data1 / np.sqrt(np.sum(data1 ** 2, 1)[:, None])

    data2 = np.random.randn(n, d + 1)
    data2 = (r * data2 / np.sqrt(np.sum(data2 ** 2, 1)[:, None])) + np.concatenate(
        (np.array([2 * r]), np.zeros(data2.shape[1] - 1)))

    data = np.concatenate((data1, data2))
    if noise:
        data += noise * np.random.randn(*data.shape)

    return torch.as_tensor(data)

# outputs persistent entropy in a specified dimension; bc: barcodes in the format of the output of gudhi, dim:
# dim: dimension of barcodes under consideration
def persistent_entropy(bc, dim):
    lt = [bc[i][1][1] - bc[i][1][0] for i in range(len(bc)) if bc[i][0] == dim and bc[i][1][1] < 10000]
    total_lt = sum(lt)
    probas = [x / total_lt for x in lt]
    logs = [-p * np.log2(p) for p in probas]
    pe = sum(logs)
    return pe