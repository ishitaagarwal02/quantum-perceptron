# import torch
from functools import reduce
# from torch.utils.data import Dataset, DataLoader
import torch
from functools import reduce
import numpy as np
import numpy as np
from tqdm import tqdm
import warnings
# from memory_profiler import profile
import gc
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader
import pdb
import os
import torch
import psutil
from memory_profiler import profile


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

complex_const = -1j

N = 10
L1 = 4
L = L1

N1 = N
j = [1] * (N1-1)
omega = -20
rabif =  [0] * (N1 - 1) + [omega]
detun = [2 * val for val in j] + [2 * sum(j)]
inter = [4 * val for val in j]


N1 = N
ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)

s = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64)
print("")


for j in range(N1):
    matrices = []
    for i in range(N1):
        m = s if i == j else p
        matrices.append(m)
    result = reduce(torch.kron, matrices)
    # matrix += rabif[j] * result
    ham.add_(rabif[j] * result)
    del result
# pdb.set_trace()

for j in range(N1):
    matrices = []
    for i in range(N1):
        m = n if i == j else p
        matrices.append(m)
    result = reduce(torch.kron, matrices)
    # matrix2 += detun[j] * result
    ham.add_(detun[j] * result)
    del result

for j in range(N1-1):
    matrices = []
    for i in range(N1-1):
        m=n if i==j else p
        matrices.append(m)
    matrices.append(n)    
    result = reduce(torch.kron, matrices)
    # matrix3 += inter[j]*result
    ham.add_(inter[j] * result)
    del result

def get_U(a,b,g):
    print(f"Memory usage in get_U: {get_memory_usage()} MB")
    gc.collect()
    N1 = N
    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64)
    identity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.complex64)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64)

    U = torch.eye(2**N1, dtype=torch.complex64)
    # iden = torch.eye(2**N1, dtype=torch.complex64)

    for i in range(4):
        pz = [pauli_z if j == i else identity for j in range(N1)]
        px = [pauli_x if j == i else identity for j in range(N1)]

        result_z = reduce(torch.kron, pz)
        result_x = reduce(torch.kron, px)

        del pz
        del px
        U = torch.matmul((torch.cos((g[i]))*torch.eye(2**N1) - 1j *torch.sin((g[i]))*result_z),torch.matmul((torch.cos((b[i]))*torch.eye(2**N1) - 1j *torch.sin((b[i]))*result_x),torch.matmul((torch.cos((a[i]))*torch.eye(2**N1) - 1j *torch.sin((a[i]))*result_z),U)))
        del result_x
        del result_z
        # del iden

    return U
def get_U2(a,b,g):
    print(f"Memory usage in get_U2: {get_memory_usage()} MB")
    gc.collect()
    N1 = N
    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64)
    identity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.complex64)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64)

    U = torch.eye(2**N1, dtype=torch.complex64)
    # iden = torch.eye(2**N1, dtype=torch.complex64)

    for i in range(4):
        pz = [pauli_z if j == i else identity for j in range(N1)]
        px = [pauli_x if j == i else identity for j in range(N1)]

        result_z = reduce(torch.kron, pz)
        result_x = reduce(torch.kron, px)

        del pz
        del px
        u1 = torch.cos(a[i])*torch.cos(b[i])*torch.cos(g[i])
        u2 = -1j* torch.cos(g[i])*torch.cos(b[i])*torch.sin(a[i])
        u3 = -1j* torch.sin(b[i])*torch.cos(g[i])*torch.cos(a[i])
        u4 = -1* torch.sin(a[i])*torch.sin(b[i])*torch.cos(g[i])
        u5 = -1j*torch.sin(g[i])*torch.cos(b[i])*torch.cos(a[i])
        u6 = -1*torch.sin(a[i])*torch.cos(b[i])*torch.sin(g[i])
        u7 = -1*torch.cos(a[i])*torch.sin(b[i])*torch.sin(g[i])
        u8 = 1j* torch.sin(a[i])*torch.sin(b[i])*torch.sin(g[i])

        U = torch.matmul((u1+u6)*torch.eye(2**N1)+ (u2+u5)*result_z+ (u3-u8)*result_x +(u4-u7)*torch.matmul(result_x,result_z),U)
        del result_x
        del result_z
    # U -= torch.eye(2**N1)
    return U


print(f"Memory usage: {get_memory_usage()} MB")

torch.manual_seed(0)
# a = torch.normal(mean=0.0, std=1., size=[L,2,N1])
# b = torch.normal(mean=0.0, std=1., size=[L,2,N1])
# g = torch.normal(mean=0.0, std=1., size=[L,2,N1])
a = torch.ones([L,2,N1])
b = torch.ones([L,2,N1])
g = torch.ones([L,2,N1])

params = (torch.stack((a,b,g)))
l = 1
# U= get_U(a[1][0],b[1][0],g[1][0])
U2= get_U2(a[1][0],b[1][0],g[1][0])
# print(U-U2)
# # print(U2)
# if torch.equal(U,U2):
#     print("Equal")
# else:
#     print("Kuch galat sa hai")

print("memory::")
def get_memory_info():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB

print(get_memory_info())
