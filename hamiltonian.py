import torch
from functools import reduce
from scipy.linalg import expm
import scipy.sparse as sparse
import scipy.linalg
import numpy as np
from scipy.sparse.linalg import expm_multiply
from graph import unit_disk_grid_graph
import numpy as np
from tqdm import tqdm

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

complex_const = -1j



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
    # time = 0.01
    Uh = ham.matrix_exp()
    return Uh

def get_U(a,b,g):
    
    N1 = 5
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
        U = u3 * u2 * u1 * U

    return U

def evolution(time, params):
    N1 = 5
    L = 4
    a = params[0]
    b = params[1]
    g = params[2]
    H = get_H(time)

    U_final = torch.eye(2**N1)

    for l in range(L):
        U1 = get_U(a[l][0],b[l][0],g[l][0])
        U2 = get_U(a[l][1],b[l][1],g[l][1])
        
        U_result = U2 * H * U1
        U_final = U_result * U_final
    
    return U_final

def expectation(state,time,params):
    N1 = 5
    U = evolution(time,params)
    state = state.reshape(4, 4)
    state = torch.flatten(torch.kron(state, torch.tensor([1.,0.])))
    # state_final = torch.abs(torch.dot(U, state)) ** 2
    state_final = torch.abs(U @ state) ** 2
    exp0 = torch.sum(state_final[::2])
    exp1 = torch.sum(state_final[1::2])
    expectation = exp0 - exp1   
    return expectation

# tensor([ 0.5508+0.0000j,  0.0000+0.0000j,  0.2456+0.2155j,  0.0000+0.0000j,
#          0.2928+0.1506j,  0.0000+0.0000j,  0.0716+0.1817j,  0.0000+0.0000j,
#          0.3332+0.1449j,  0.0000+0.0000j,  0.0919+0.1950j,  0.0000+0.0000j,
#          0.1375+0.1681j,  0.0000+0.0000j, -0.0045+0.1288j, -0.0000+0.0000j,
#          0.2108+0.1786j,  0.0000+0.0000j,  0.0241+0.1621j,  0.0000+0.0000j,
#          0.0632+0.1526j,  0.0000+0.0000j, -0.0315+0.0928j, -0.0000+0.0000j,
#          0.0806+0.1635j,  0.0000+0.0000j, -0.0281+0.1044j, -0.0000+0.0000j,
#         -0.0019+0.1089j, -0.0000+0.0000j, -0.0435+0.0478j, -0.0000+0.0000j],
#        dtype=torch.complex128, grad_fn=<ReshapeAliasBackward0>)
# L = 4
# N1 = 5

# a = torch.normal(mean=0, std=1, size=[L,2,N1])
# b = torch.normal(mean=0, std=1, size=[L,2,N1])
# g = torch.normal(mean=0, std=1, size=[L,2,N1])
# r = []
# params = [a,b,g]



# for t in torch.arange(0.01,0.1,0.01):
#     r.append(expectation(state, t, params))
# r.append(1.)


#####################  Import Data loader ###########################

from heisenberg import QuantumDataset
from mfa import MFADataset, MFADatasetLoader

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pdb

import torch.autograd.profiler as profiler

import torch.nn as nn
import torch.optim as optim

GLOBAL_LIST = []


num_layers = 2

class QuantumPerceptron(nn.Module):
    """"
    mfaPerceptron
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(QuantumPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.mid_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mid_layers.append(nn.Linear(hidden_size, hidden_size))
        self.layer2 = nn.Linear(hidden_size, output_size)
        L = 4
        N1 = 5
        a = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        b = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        g = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        self.params = nn.Parameter(torch.stack((a,b,g)))
        self.params.requires_grad = True

        # self.params = nn.Parameter(torch.tensor([1.,0.,0.,0.,0.,0.,0.,0]))

    def init_r(self, state):
        self.r = []
        for t in torch.arange(0.01,0.1,0.01):
            self.r.append(expectation(state, t, self.params))
        self.r.append(torch.tensor(1.).to(torch.float32))
        self.r = torch.stack(self.r)
        return self.r

    def forward(self, state):
        r = self.init_r(state)
        self.r = self.r.to(torch.float32)
        r = r.reshape(1, -1)
        print(self.params)
        # self.r = state.to(torch.float32)
        # self.r = torch.tensor(r).to(torch.float32)
        # state = torch.tensor(state).reshape(1,16)
        # self.r = state.to(torch.float32)
        print(self.r)
        out = self.layer1(self.r)
        out = F.gelu(out)
        for layer in self.mid_layers:
            out = layer(out)
            out = F.gelu(out)
        out = self.layer2(out)
        out = F.tanh(out)
        # return return_energy(out)
        return out
        # return out

# L = 4
# N1 = 5
# input_size = 2**N1  # number of parameters in the return_energy function
# output_size = 2*L*N1*3  # since return_energy returns a single value

# a = torch.normal(mean=0, std=1, size=[L,2,N1])
# b = torch.normal(mean=0, std=1, size=[L,2,N1])
# g = torch.normal(mean=0, std=1, size=[L,2,N1])
# r = []
# params = [a,b,g]

# for t in torch.arange(0.01,0.1,0.01):
#     r.append(expectation(state, t, params))
# r.append(1.)


model = QuantumPerceptron(input_size= 10, output_size= 1, hidden_size = 144)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
model = model
# params = torch.randn(8)  # initial 
# Initialize parameters with Gaussian distribution centered at 0
import matplotlib.pyplot as plt
# Set a seed for reproducibility
torch.manual_seed(0)

x = QuantumDatasetLoader()
y = MFADatasetLoader()

heisenberg_dataset, heisenberg_dataloader = x.return_dataset()
mfa_dataset, mfa_dataloader = y.return_dataset()

import pandas as pd

# Assume we have a DataLoader named 'dataloader'

# Create lists for each column in the DataFrame
Js = []
eigenvalues = []
eigenvectors = []
vals = []

# # Iterate over the DataLoader
# for J, eigenvalue, eigenvector, val in tqdm(heisenberg_dataloader):
#     # Add the values to their respective lists
#     Js.append(J)
#     eigenvalues.append(eigenvalue)
#     eigenvectors.append(eigenvector)
#     vals.append(val)

# for J, eigenvalue, _, val, eigenvector in tqdm(mfa_dataloader):
#     # Add the values to their respective lists
#     Js.append(J)
#     eigenvalues.append(eigenvalue)
#     eigenvectors.append(eigenvector)
#     vals.append(val)

# Convert lists into a DataFrame
# df = pd.DataFrame({
#     'J': Js,
#     'Eigenvalue': eigenvalues,
#     'Eigenvector': eigenvectors,
#     'Val':vals,
# })



for epoch in range(500):
    losses = []
 
    
    model.train()
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)



    # for mfa, quant in tqdm(zip(mfa_dataloader, heisenberg_dataloader)):
    for J, eigenvalue, eigenvector, val in tqdm(heisenberg_dataloader):
        pred = model(eigenvector)
        loss = criterion(pred, val.float())
        total_loss += loss
        losses.append(loss.item())
        
    for J, eigenvalue, _, val, eigenvector in tqdm(mfa_dataloader):
        eigenvector = eigenvector.to(torch.complex128)
        eigenvector = eigenvector.reshape(1, -1)
        eigenvector = eigenvector.detach()
        val = val.detach()
        pred = model(eigenvector)
        loss = criterion(pred, val.float())
        total_loss += loss 
        losses.append(loss.item())
    #`pdb.set_trace()
    print("Iterating through Heisenberg...")
    total_loss.backward()
    print(total_loss)

    for param in model.parameters():
        param.data.clamp_(0, 2*torch.pi)

    #     _, _, _, val, eigenvector = mfa
    #     pred = model(eigenvector)
    #     loss = criterion(pred, val.float())
    #     total_loss += loss

    #     J, eigenvalue, eigenvector, val = quant
    #     pred = model(eigenvector)
    #     loss = criterion(pred, val.float())
    #     total_loss += loss

    # pdb.set_trace()

    optimizer.step()
    optimizer.zero_grad()

    # Clip the parameters to be within the bounds


    
    # if epoch % 1 == 0:
    #     print("Loss: {} || last_prediction: {} || epoch: {}/500".format(total_loss, pred, epoch))
    #     plt.plot([loss.detach().numpy() for loss in losses])
    #     print("Parameters: ")
        # for name, param in model.named_parameters():
        #     print("{}: {}".format(name, param.data))

    model.eval()

# We don't need to track gradients for validation, so wrap in 
# no_grad to save memory

    if epoch % 5 == 0 and epoch != 0:
        print("Starting accuracy calculation...")
        with torch.no_grad():

            correct_predictions = 0
            total_predictions = 0

            print("Iterating through Heisenberg (for accuracy)...")    
            for J, eigenvalue, eigenvector, val in tqdm(heisenberg_dataloader):
                pred = model(eigenvector)
                # Convert predictions and true values to -1 or 1
                pred_rounded = torch.where(pred < 0, -1, 1)
                val_rounded = torch.where(val > 0., 1, -1)
                # pdb.set_trace()

                # Count correct predictions
                correct_predictions += torch.sum(pred_rounded == val_rounded).item()

                total_predictions += len(val)

            print("Iterating through MFA (for accuracy)...")
            for J, eigenvalue, _, val, eigenvector in tqdm(mfa_dataloader):
                pred = model(eigenvector)

                # Convert predictions and true values to -1 or 1
                pred_rounded = torch.where(pred > 0.5, 1, 0)
                val_rounded = torch.where(val == 1., 1, 0)

                # Count correct predictions
                correct_predictions += torch.sum(pred_rounded == val_rounded).item()

                total_predictions += len(val)

            accuracy = correct_predictions / total_predictions
            print(f'Accuracy: {accuracy}')



torch.save(model)











