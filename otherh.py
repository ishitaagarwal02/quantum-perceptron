import torch
from functools import reduce
from graph import unit_disk_grid_graph
import numpy as np

N1 = 3
print(N1)

matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

s = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)

rabif = [0,0,0,1]
detun = []
J = [1, 2, 3]

for j in range(N1):
    matrices = []
    for i in range(N1):
        m = s if i == j else p
        matrices.append(m)
    result = reduce(torch.kron, matrices)
    matrix += rabif[j] * result

for j in range(N1):
    matrices = []
    for i in range(N1):
        m = n if i == j else p
        matrices.append(m)
    result = reduce(torch.kron, matrices)
    matrix2 += detun[j] * result

for j in range(1,N1):
    for k in range(j):
        print([k,j])
        matrices = []
        for i in range(N1):
            m=J[i]*s if i==k or i==j else p
            matrices.append(m)
        result = reduce(torch.kron, matrices)
        matrix3 += result

ham = 1/2*matrix - matrix2 + matrix3
print(ham)









