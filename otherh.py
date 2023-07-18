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

import torch

def get_H(time):
        
    j = [1,1,1,1]
    omega = -10
    rabif = [0,0,0,0,omega]
    detun = [2*j[0], 2*j[1], 2*j[2], 2*j[3], 2*(j[0]+j[1]+j[2]+j[3])]
    inter = [4*j[0], 4*j[1], 4*j[2], 4*j[3]] 
    #length of inter would be number of edges and would be labelled based on temp

    N1 = 5
    # Initialize the resulting matrix as zero
    matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

    s = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
    n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)

    # Loop over all combinations
    for j in range(N1):
        matrices = []
        for i in range(N1):
            m = s if i == j else p
            matrices.append(m)
        result = reduce(torch.kron, matrices)
        matrix += rabif[j] * result
    # pdb.set_trace()

    for j in range(N1):
        matrices = []
        for i in range(N1):
            m = n if i == j else p
            matrices.append(m)
        result = reduce(torch.kron, matrices)
        matrix2 += detun[j] * result
    print(matrix2)
    for j in range(N1-1):
        matrices = []
        for i in range(N1-1):
            m=n if i==j else p
            matrices.append(m)
        matrices.append(n)    
        result = reduce(torch.kron, matrices)
        matrix3 += inter[j]*result
    print(matrix3)
    complex_const = -1j
    time = 0.01

    ham = (complex_const * time)*(matrix - matrix2 + matrix3)
    # if ham.is_cuda:
    #     ham = ham.cpu()
    # ham_np = ham.numpy()
    # time = 0.01
    Uh = ham.matrix_exp()
    return Uh


UH=get_H(0.01)
print(UH)




