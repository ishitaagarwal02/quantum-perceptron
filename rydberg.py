import torch
from functools import reduce
import numpy as np


# rabif = [1,1,1,1]
# detun = [2,2,2,2]
# inter = [3,3,3] 
# #length of inter would be number of edges and would be labelled based on temp

# N1 = 4
# print(N1)

# Initialize the resulting matrix as zero
# matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
# matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
# matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
# ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

# s = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
# p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
# n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)

# # Loop over all combinations
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

# for j in range(N1-1):
#     matrices = []
#     for i in range(N1):
#         m=n if i==j or i==j+1 else p
#         matrices.append(m)
#     result = reduce(torch.kron, matrices)
#     matrix3 += inter[j] * result

# ham = matrix - matrix2 + matrix3
# print(ham)

N1 = 3
pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
identity = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

u1 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
u2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
u3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
U = torch.eye(2**N1, dtype=torch.complex128)
U1 = torch.eye(2**N1, dtype=torch.complex128)


# a = [1.,1.,1.,1.,1.]
# b = [2.,2.,2.,2.,2.]
# g = [1.,1.,1.,1.,1.]
a = [1.,1.,1.]
b = [1.,1.,1.]
g = [1.,1.,1.]
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
    # U1 = u3 * u2 * u1 * U1
    u1 = torch.cos(torch.tensor(a[i]))*torch.eye(2**N1) - 1j *torch.sin(torch.tensor(a[i]))*result_z
    u2 = torch.cos(torch.tensor(b[i]))*torch.eye(2**N1) - 1j *torch.sin(torch.tensor(b[i]))*result_x
    u3 = torch.cos(torch.tensor(g[i]))*torch.eye(2**N1) - 1j *torch.sin(torch.tensor(g[i]))*result_z
    U = u3 * u2 * u1 * U
    # print(U)

print(U1)
print(U)










