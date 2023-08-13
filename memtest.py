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

    for i in range(N1):
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
@profile
def evolution(time, params):

    global ham
    # gc.collect()
    N1 = N
    L = L1
    print(L)
    a = params[0]
    b = params[1]
    g = params[2]
    complex_const = -1j
    ham = (complex_const * time)*ham
    # pdb.set_trace()
    # if ham.is_cuda:
    #     ham = ham.cpu()
    # ham_np = ham.numpy()
    # time = 0.01
    H = ham.matrix_exp()
    # H = get_H(time)

    U_final = torch.eye(2**N1, dtype=torch.complex64)
    for l in range(L):        
        U1 = get_U(a[l][0],b[l][0],g[l][0])
        print(f"Memory usage afterU1: {get_memory_usage()} MB")

        print("here")
        U2 = get_U(a[l][1],b[l][1],g[l][1])
        U_final = torch.matmul(torch.matmul(torch.matmul(U2, H), U1), U_final)
        del U1
        del U2

    return U_final

@profile
def expectation(state,time,params):
    N1 = N
    # pdb.set_trace()
    L = L1
    state = (torch.kron(state, torch.tensor([1.,0.])))
    # for l in range(L):
    U = evolution(time,params)
    # pdb.set_trace()
    print(f"Memory usage in expectation: {get_memory_usage()} MB")
    # state = U @ state.view(-1,state.size()[0])
    state = U @ state
    del U
    state_final = torch.abs(state) ** 2
    exp0 = torch.sum(state_final[::2], axis = 0)
    exp1 = torch.sum(state_final[1::2], axis = 0)
    del state_final
    expt = exp0 - exp1
    print(f"Memory usage after calculating expt: {get_memory_usage()} MB")

    return expt


print(f"Memory usage: {get_memory_usage()} MB")

a = torch.normal(mean=0.0, std=1., size=[L,2,N1])
b = torch.normal(mean=0.0, std=1., size=[L,2,N1])
g = torch.normal(mean=0.0, std=1., size=[L,2,N1])
params = (torch.stack((a,b,g)))
# l = 1
# U= get_U(a[1][0],b[1][0],g[1][0])
def z2phase():
    
    N1 = 9
    states10 = []
    states01 = []
    zero_state = torch.tensor([1, 0], dtype=torch.complex64)
    one_state = torch.tensor([0, 1], dtype=torch.complex64)
    matrix = []
    # for i in torch.arange(15):
        # generate 0101 type states
    for j in torch.arange(N1):
        r = torch.rand(1) * (1. - 0.7) + 0.7
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # e_phi = torch.cos(phi) + 1j*torch.sin(phi)
        # e_phi = torch.tensor(e_phi) 
        if j%2==0:
            s = torch.sqrt(r)*zero_state + torch.sqrt(1-r)*one_state
        else:
            s = torch.sqrt(1-r)*zero_state + torch.sqrt(r)*one_state
        matrix.append(s)
    
    states10 = reduce(torch.kron, matrix)
    matrix = []
    # generate 1010 type states
    for j in torch.arange(N1):
        r = torch.rand(1) * (1. - 0.7) + 0.7
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # e_phi = torch.cos(phi) + 1j*torch.sin(phi)
        # e_phi = torch.tensor(e_phi) 
        if j%2==1:
            s = torch.sqrt(r)*zero_state + torch.sqrt(1-r)*one_state
        else:
            s = torch.sqrt(1-r)*zero_state + torch.sqrt(r)*one_state
        # pdb.set_trace()
        matrix.append(s)
        
    states01 = reduce(torch.kron, matrix) 
    return states10, states01
s21, s22 = z2phase()

expt = expectation(s21, 0.01,params)

print("memory::")
def get_memory_info():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB

print(get_memory_info())
