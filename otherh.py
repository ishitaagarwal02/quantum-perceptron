import torch
from functools import reduce
from graph import unit_disk_grid_graph
import numpy as np
import pdb
# N1 = 3
# print(N1)

# matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
# matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
# matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
# ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

# s = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
# p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
# n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)

# rabif = [0,0,0,1]
# detun = []
# J = [1, 2, 3]

# for j in range(N1):
#     matrices = []
#     for i in range(N1):
#         m = s if i == j else p
#         matrices.append(m)
#     result = reduce(torch.kron, matrices)
#     matrix += rabif[j] * result

# for j in range(N1):
#     matrices = []
#     for i in range(N1):
#         m = n if i == j else p
#         matrices.append(m)
#     result = reduce(torch.kron, matrices)
#     matrix2 += detun[j] * result

# for j in range(1,N1):
#     for k in range(j):
#         print([k,j])
#         matrices = []
#         for i in range(N1):
#             m=J[i]*s if i==k or i==j else p
#             matrices.append(m)
#         result = reduce(torch.kron, matrices)
#         matrix3 += result

# ham = 1/2*matrix - matrix2 + matrix3
# print(ham)

# from heisenberg import QuantumDataset, QuantumDatasetLoader
# from mfa import MFADataset, MFADatasetLoader

# from torch.utils.data import Dataset, DataLoader
# import pdb

# complex_const = -1j

# import torch

# def get_H(time):
        
#     j = [1,1,1,1]
#     omega = -10
#     rabif = [0,0,0,0,omega]
#     detun = [2*j[0], 2*j[1], 2*j[2], 2*j[3], 2*(j[0]+j[1]+j[2]+j[3])]
#     inter = [4*j[0], 4*j[1], 4*j[2], 4*j[3]] 
#     #length of inter would be number of edges and would be labelled based on temp

#     N1 = 5
#     # Initialize the resulting matrix as zero
#     matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
#     matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
#     matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
#     ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

#     s = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
#     p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
#     n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)

#     # Loop over all combinations
#     for j in range(N1):
#         matrices = []
#         for i in range(N1):
#             m = s if i == j else p
#             matrices.append(m)
#         result = reduce(torch.kron, matrices)
#         matrix += rabif[j] * result
#     # pdb.set_trace()

#     for j in range(N1):
#         matrices = []
#         for i in range(N1):
#             m = n if i == j else p
#             matrices.append(m)
#         result = reduce(torch.kron, matrices)
#         matrix2 += detun[j] * result
#     print(matrix2)
#     for j in range(N1-1):
#         matrices = []
#         for i in range(N1-1):
#             m=n if i==j else p
#             matrices.append(m)
#         matrices.append(n)    
#         result = reduce(torch.kron, matrices)
#         matrix3 += inter[j]*result
#     print(matrix3)
#     complex_const = -1j
#     time = 0.01

#     ham = (complex_const * time)*(matrix - matrix2 + matrix3)
#     # if ham.is_cuda:
#     #     ham = ham.cpu()
#     # ham_np = ham.numpy()
#     # time = 0.01
#     Uh = ham.matrix_exp()
#     return Uh


# UH=get_H(0.01)
# print(UH)


def get_U(a,b,g):
    
    N1 = 2
    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
    identity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.complex128)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128)

    # u1 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    # u2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    # u3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    U = torch.eye(2**N1, dtype=torch.complex128)

    for i in range(N1):
        pz = []
        px = []
        for j in range(N1):
            z = pauli_z if j == i else identity
            x = pauli_x if j == i else identity
            pz.append(z)
            px.append(x)
        result_z = reduce(torch.kron, pz)
        result_x = reduce(torch.kron, px)
        # u1 = torch.tensor(-1j * a[i] * result_z).matrix_exp()
        # u2 = torch.tensor(-1j * b[i] * result_x).matrix_exp()
        # u3 = torch.tensor(-1j * g[i] * result_z).matrix_exp()
        u1 = torch.cos((a[i]))*torch.eye(2**N1) - 1j *torch.sin((a[i]))*result_z
        u2 = torch.cos((b[i]))*torch.eye(2**N1) - 1j *torch.sin((b[i]))*result_x
        u3 = torch.cos((g[i]))*torch.eye(2**N1) - 1j *torch.sin((g[i]))*result_z
        U = u3 @ u2 @ u1 @ U

    return U

from functools import reduce
def single_qubit_gate(angle, axis):
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    
    if axis == "x":
        return torch.tensor([[cos_theta, -1j * sin_theta],
                             [-1j * sin_theta, cos_theta]], dtype=torch.complex128)
    elif axis == "y":
        return torch.tensor([[cos_theta, -sin_theta],
                             [sin_theta, cos_theta]], dtype=torch.complex128)
    elif axis == "z":
        return torch.tensor([[torch.exp(-1j * angle), 0],
                             [0, torch.exp(1j * angle)]], dtype=torch.complex128)
    else:
        raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

# Example usage
N1 = 2
a = torch.tensor([0.1, 0.2], dtype=torch.float64)  # angles for z-axis rotation
b = torch.tensor([0.3, 0.4], dtype=torch.float64)  # angles for x-axis rotation
g = torch.tensor([0.5, 0.6], dtype=torch.float64)  # angles for z-axis rotation

U = torch.eye(2 ** N1, dtype=torch.complex128)  # Initialize U as the identity matrix

for i in range(N1):
    result_z = torch.eye(2, dtype=torch.complex128)  # Initialize result_z as 2x2 identity matrix
    result_x = torch.eye(2, dtype=torch.complex128)  # Initialize result_x as 2x2 identity matrix
    
    for j in range(N1):
        if j == i:
            result_z = torch.kron(result_z, single_qubit_gate(a[i], "z"))
            result_x = torch.kron(result_x, single_qubit_gate(b[i], "x"))
        else:
            result_z = torch.kron(result_z, torch.eye(2, dtype=torch.complex128))
            result_x = torch.kron(result_x, torch.eye(2, dtype=torch.complex128))
    
    u1 = torch.cos(a[i]) * torch.eye(2 ** N1, dtype=torch.complex128) - 1j * torch.sin(a[i]) * result_z
    u2 = torch.cos(b[i]) * torch.eye(2 ** N1, dtype=torch.complex128) - 1j * torch.sin(b[i]) * result_x
    u3 = torch.cos(g[i]) * torch.eye(2 ** N1, dtype=torch.complex128) - 1j * torch.sin(g[i]) * result_z
    U = u3 @ u2 @ u1 @ U

print(U)

U=get_U(a,b,g)
print(U)