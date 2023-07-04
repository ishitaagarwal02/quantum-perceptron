import torch
import pdb
from functools import reduce
from scipy.linalg import expm
import scipy.sparse as sparse
import scipy.linalg
import numpy as np
from scipy.sparse.linalg import expm_multiply
from graph import unit_disk_grid_graph
import numpy as np

def heisenberg(J):
    N1 = 4
    # print(N1)

    # Initialize the resulting matrix as zero
    matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

    p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    # J = -1

    # Loop over all combinations
    for j in range(N1-1):
        matrices = []
        for i in range(N1):
            m = s1 if i == j or i==(j+1) else p
            matrices.append(m)
        # pdb.set_trace()
        result = reduce(torch.kron, matrices)
        matrix += result

    for j in range(N1-1):
        matrices = []
        for i in range(N1):
            m = s2 if i == j or i==j+1 else p
            matrices.append(m)
        # print(matrices)
        # pdb.set_trace()
        result = reduce(torch.kron, matrices)
        matrix2 += result

    for j in range(N1-1):
        matrices = []
        for i in range(N1):
            m=s3 if i==j or i==j+1 else p
            matrices.append(m)
        result = reduce(torch.kron, matrices)
        # pdb.set_trace()
        matrix3 += result

    ham = J*(matrix + matrix2 + matrix3)
    eigenvalues, eigenvectors = torch.linalg.eigh(ham)
    eigenvectors = eigenvectors.t()
    threshold = 1e-10
    eigenvectors = torch.where(abs(eigenvectors) < threshold, torch.tensor(0.0), eigenvectors)
    # print(ham)
    # print(eigenvalues)
    # pdb.set_trace
    # print(eigenvectors)
    return eigenvalues[0], eigenvectors[0]

# with open("C:\\Users\\91833\\Desktop\\SURF\\output_gs.txt", "w") as file:
for J in torch.arange(-1, 1.01, 0.1):
    eigenvalue, eigenvector = heisenberg(J)
    print(J, eigenvalue, eigenvector, -1)

from torch.utils.data import Dataset, DataLoader

# Define a custom dataset
class QuantumDataset(Dataset):
    def __init__(self, J_values):
        self.J_values = J_values

    def __len__(self):
        return len(self.J_values)

    def __getitem__(self, idx):
        J = self.J_values[idx]
        eigenvalue, eigenvector = heisenberg(J)
        return J, eigenvalue, eigenvector, -1

# Create a PyTorch DataLoader
J_values = torch.arange(-1, 1.01, 0.1)


class QuantumDatasetLoader():
    def return_dataset(self):
        dataset = QuantumDataset(J_values)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataset, dataloader

# Iterate over the dataloader
# for J, eigenvalue, eigenvector, val in dataloader:
#     print(f"J: {J.item():.2f} | Eigenvalue: {eigenvalue.item():.4f} | Eigenvector: {eigenvector.squeeze()} | Value: {val.item()}")
    


# for J in torch.arange(-1, 1.01, 0.01):
#     ham = J*(matrix + matrix2 + matrix3)
#     eigenvalues, eigenvectors = torch.linalg.eigh(ham)
#     sorted_indices = torch.argsort(eigenvalues)
#     eigenvalues = eigenvalues[sorted_indices]
#     eigenvectors = eigenvectors[:, sorted_indices]
#     gs.append((J, eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()))
# print(gs[0][2])

# vec0 = np.zeros(2**N1)
# vec0[2] = 1
# if ham.is_cuda:
#     ham = ham.cpu()
# ham_np = ham.numpy()

# complex_const = -1j
# time = 0.01
# U = scipy.linalg.expm(complex_const * ham_np * time)
# vec1 = np.abs(np.dot(U, vec0)) ** 2

# print(vec1)