import torch
from functools import reduce
from scipy.linalg import expm
import scipy.sparse as sparse
import scipy.linalg
import numpy as np
from scipy.sparse.linalg import expm_multiply
from graph import unit_disk_grid_graph
import numpy as np

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

from heisenberg import QuantumDataset, QuantumDatasetLoader
from mfa import MFADataset, MFADatasetLoader

from torch.utils.data import Dataset, DataLoader
import pdb
pdb.set_trace()



def get_H(time):
        
    j = [1,1,1,1]
    omega = -50
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

    for j in range(N1):
        matrices = []
        for i in range(N1):
            m = n if i == j else p
            matrices.append(m)
        result = reduce(torch.kron, matrices)
        matrix2 += detun[j] * result

    for j in range(N1-1):
        matrices = []
        for i in range(N1-1):
            m=n if i==j else p
            matrices.append(m)
        matrices.append(n)    
        result = reduce(torch.kron, matrices)
        matrix3 += inter[j]*result

    ham = (complex_const * time)*(matrix - matrix2 + matrix3)
    # if ham.is_cuda:
    #     ham = ham.cpu()
    # ham_np = ham.numpy()
    complex_const = -1j
    time = 0.01
    Uh = ham.matrix_exp()
    return Uh

def get_U(a,b,g):
    
    N1 = 5
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    identity = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    u1 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    u2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    u3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
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
        u1 = torch.tensor(-1j * a[i] * result_z).matrix_exp()
        u2 = torch.tensor(-1j * b[i] * result_x).matrix_exp()
        u3 = torch.tensor(-1j * g[i] * result_z).matrix_exp()
        U = u3 * u2 * u1 * U

    return U

def evolution(time, params):
    N1 = 5
    L = 4
    # l=4, k=1,2 and i goes from 1 to N1
    # a = torch.ones([L,2,N1])
    # b = torch.ones([L,2,N1])
    # g = torch.ones([L,2,N1])
    a = params[0]
    b = params[1]
    c = params[2]
    H = get_H(time)

    for l in range(L):
        U1 = get_U(a[l][0],b[l][0],g[l][0])
        U2 = get_U(a[l][1],b[l][1],g[l][1])
        
        U_result = U2 * H * U1
        U_final = U_result * U_final
    
    return U_final

def expectation(state,time,params):
    N1 = 5
    U = evolution(time,params)
    state_final = np.abs(np.dot(U, state)) ** 2
    exp0 = torch.sum(state_final[::2])
    exp1 = torch.sum(state_final[1::2])
    expectation = exp0 - exp1   
    return expectation


L = 4
N1 = 5

a = torch.normal(mean=0, std=1, size=[L,2,N1])
b = torch.normal(mean=0, std=1, size=[L,2,N1])
g = torch.normal(mean=0, std=1, size=[L,2,N1])
r = []
params = [a,b,g]

for t in torch.arange(0.01,0.1,0.01):
    r.append(expectation(state, t, params))
r.append(1.)


#####################  Import Data loader ###########################

from heisenberg import QuantumDataset
from mfa import MFADataset, MFADatasetLoader

from torch.utils.data import Dataset, DataLoader
import pdb
pdb.set_trace()









import torch.nn as nn
import torch.optim as optim

class mfaPerceptron(nn.Module):
    """"
    mfaPerceptron
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(mfaPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

        # self.params = nn.Parameter(torch.tensor([1.,0.,0.,0.,0.,0.,0.,0]))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # return return_energy(out)
        return self.return_energy(out), out
        # return out

L = 4
N1 = 5
input_size = 2**N1  # number of parameters in the return_energy function
output_size = 2*L*N1*3  # since return_energy returns a single value

a = torch.normal(mean=0, std=1, size=[L,2,N1])
b = torch.normal(mean=0, std=1, size=[L,2,N1])
g = torch.normal(mean=0, std=1, size=[L,2,N1])
r = []
params = [a,b,g]

for t in torch.arange(0.01,0.1,0.01):
    r.append(expectation(state, t, params))
r.append(1.)

model = mfaPerceptron(input_size= input_size, output_size= output_size, hidden_size = 16)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# params = torch.randn(8)  # initial 
# Initialize parameters with Gaussian distribution centered at 0

# Set a seed for reproducibility
torch.manual_seed(0)

for epoch in range(500):
    params = params.requires_grad_()  # make sure gradients are computed with respect to params
    #`pdb.set_trace()
    outputs, out = model(x)
    # target = torch.tensor([-3.0])  # target energy is -3
    # loss = criterion(outputs, target)
    # print(loss)

    optimizer.zero_grad()
    outputs.backward()
    optimizer.step()
    # Clip the parameters to be within the bounds
    for param in model.parameters():
        param.data.clamp_(0, 2 * pi)

    if epoch % 25 == 0:
        print("{} || {} || {}/500".format(out, outputs, epoch))
        print("Parameters: ")
        # for name, param in model.named_parameters():
        #     print("{}: {}".format(name, param.data))












