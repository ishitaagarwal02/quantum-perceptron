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
import torch
from functools import reduce
import numpy as np
import numpy as np
from tqdm import tqdm
import warnings
# from memory_profiler import profile
warnings.filterwarnings("ignore")
import tracemalloc

# @profile
# print(ham)
# vec0 = np.zeros(2**N1)
# vec0[0] = 1
# # if ham.is_cuda:
# #     ham = ham.cpu()
# # ham_np = ham.numpy()

# # complex_const = -1j
# # time = 0.01
# # U = scipy.linalg.expm(complex_const * ham_np * time)
# vec1 = np.abs(np.dot(U, vec0)) ** 2

# print(vec1)
# device = "cuda:0"

from torch.utils.data import Dataset, DataLoader
import pdb

complex_const = -1j

tracemalloc.start()

j = [1,1,1,1,1]
omega = -20
rabif = [0,0,0,0,0,omega]
detun = [2*j[0], 2*j[1], 2*j[2], 2*j[3], 2*j[4], 2*(j[0]+j[1]+j[2]+j[3]+j[4])]
inter = [4*j[0], 4*j[1], 4*j[2], 4*j[3], 4*j[4]] 

# j = [1,1,1,1,1,1,1,1]
# omega = -20
# rabif = [0,0,0,0,0,0,0,0,omega]
# detun = [2*j[0], 2*j[1], 2*j[2], 2*j[3], 2*j[4], 2*j[5], 2*j[6], 2*j[7], 2*(j[0]+j[1]+j[2]+j[3]+j[4]+j[5]+j[6]+j[7])]
# inter = [4*j[0], 4*j[1], 4*j[2], 4*j[3], 4*j[4], 4*j[5], 4*j[6], 4*j[7]]


N1 = 6
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
    # matrix += rabif[j] * result
    matrix.add_(rabif[j] * result)
    del result
# pdb.set_trace()

for j in range(N1):
    matrices = []
    for i in range(N1):
        m = n if i == j else p
        matrices.append(m)
    result = reduce(torch.kron, matrices)
    # matrix2 += detun[j] * result
    matrix2.add_(detun[j] * result)
    del result

for j in range(N1-1):
    matrices = []
    for i in range(N1-1):
        m=n if i==j else p
        matrices.append(m)
    matrices.append(n)    
    result = reduce(torch.kron, matrices)
    # matrix3 += inter[j]*result
    matrix3.add_(inter[j] * result)
    del result


def get_U(a,b,g):
    
    N1 = 6
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
        del pz
        del px
        # u1 = torch.tensor(-1j * a[i] * result_z).matrix_exp()
        # u2 = torch.tensor(-1j * b[i] * result_x).matrix_exp()
        # u3 = torch.tensor(-1j * g[i] * result_z).matrix_exp()
        u1 = torch.cos((a[i]))*torch.eye(2**N1) - 1j *torch.sin((a[i]))*result_z
        u2 = torch.cos((b[i]))*torch.eye(2**N1) - 1j *torch.sin((b[i]))*result_x
        u3 = torch.cos((g[i]))*torch.eye(2**N1) - 1j *torch.sin((g[i]))*result_z
        del result_x
        del result_z
        U = u3 @ u2 @ u1 @ U
        del u1
        del u2
        del u3
    print("what")
    # pdb.set_trace()
    print(tracemalloc.get_traced_memory())

    # U = torch.matmul((torch.cos((g[i]))*torch.eye(2**N1) - 1j *torch.sin((g[i]))*result_z),torch.matmul((torch.cos((b[i]))*torch.eye(2**N1) - 1j *torch.sin((b[i]))*result_x),torch.matmul((torch.cos((a[i]))*torch.eye(2**N1) - 1j *torch.sin((a[i]))*result_z),U)))
    
    return U

def evolution(time, params):
    N1 = 6
    L = 1
    a = params[0]
    b = params[1]
    g = params[2]
    complex_const = -1j
    ham = (complex_const * time)*(matrix - matrix2 + matrix3)
    # pdb.set_trace()
    # if ham.is_cuda:
    #     ham = ham.cpu()
    # ham_np = ham.numpy()
    # time = 0.01
    H = ham.matrix_exp()
    # H = get_H(time)

    U_final = torch.eye(2**N1, dtype=torch.complex128)
    # U1 = get_U(a[l][0],b[l][0],g[l][0])
    # U2 = get_U(a[l][1],b[l][1],g[l][1])
    # pdb.set_trace()
    # U_result = torch.matmul(torch.matmul(U2, H), U1)
    # U_final = torch.matmul(U_result, U_final)
    for l in range(L):
        print("calling U")
        U1 = get_U(a[l][0],b[l][0],g[l][0])
        print("here")
        U2 = get_U(a[l][1],b[l][1],g[l][1])
        # pdb.set_trace()
        # U_result = torch.matmul(torch.matmul(U2, H), U1)
        # U_final = torch.matmul(U_result, U_final)
        U_final = torch.matmul(torch.matmul(torch.matmul(U2, H), U1), U_final)

    return U_final

def expectation(state,time,params):
    N1 = 6
    # pdb.set_trace()
    L = 1
    # state = state.reshape(4, 4)
    # state = torch.flatten(torch.kron(state, torch.tensor([1.,0.])))
    # state = state.to(device)
    state = (torch.kron(state, torch.tensor([1.,0.])))
    # for l in range(L):
    U = evolution(time,params)
    state = U @ state.view(-1,state.size()[0])
    state_final = torch.abs(state) ** 2
    exp0 = torch.sum(state_final[::2], axis = 0)
    exp1 = torch.sum(state_final[1::2], axis = 0)
    expectation = exp0 - exp1
    return state_final, expectation
L = 1
a = torch.normal(mean=0.0, std=1., size=[L,2,N1])
b = torch.normal(mean=0.0, std=1., size=[L,2,N1])
g = torch.normal(mean=0.0, std=1., size=[L,2,N1])
params = torch.stack((a,b,g))
U = evolution(0.01, params, )
tracemalloc.stop()