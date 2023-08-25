import torch
torch.cuda.empty_cache()
from functools import reduce
import numpy as np
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

device = "cuda:0"

from torch.utils.data import Dataset, DataLoader
import pdb

complex_const = -1j


N = 9
L1 = 5

N1 = N
j = [torch.normal(0., 1., size=(1,)).item()] * (N1-1)
omega = -10
rabif =  [0] * (N1 - 1) + [omega]
detun = [2 * val for val in j] + [2 * sum(j)]
inter = [4 * val for val in j]

# Initialize the resulting matrix as zero
# matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)
# matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)
# matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)
ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex64).to(device)

s = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64).to(device)
p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64).to(device)
n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64).to(device)

# matrix = matrix.to(device)
# matrix2 = matrix2.to(device)
# matrix3 = matrix3.to(device)
# ham = ham.to(device)
# s = s.to(device)
# p = p.to(device)
# n = n.to(device)

# Loop over all combinations
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
    ham.add_(-1 * detun[j] * result)
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
    
    N1 = N
    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64).to(device)
    identity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.complex64).to(device)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64).to(device)
    # pauli_x = pauli_x.to(device)
    # identity = identity.to(device)
    # pauli_z = pauli_z.to(device)

    # u1 = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)
    # u2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)
    # u3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)
    U = torch.eye(2**N1, dtype=torch.complex64).to(device)
    # U = U.to(device)
    # iden = torch.eye(2**N1).to(device)

    for i in range(N1):
        pz = [pauli_z if j == i else identity for j in range(N1)]
        px = [pauli_x if j == i else identity for j in range(N1)]

        result_z = reduce(torch.kron, pz).to(device)
        result_x = reduce(torch.kron, px).to(device)
        # result_z = result_z.to(device)
        # result_x = result_x.to(device)
        del px
        del pz
        # u1 = torch.tensor(-1j * a[i] * result_z).matrix_exp()
        # u2 = torch.tensor(-1j * b[i] * result_x).matrix_exp()
        # u3 = torch.tensor(-1j * g[i] * result_z).matrix_exp()
        # u1 = torch.cos((a[i]))*iden - 1j *torch.sin((a[i]))*result_z
        # u2 = torch.cos((b[i]))*iden - 1j *torch.sin((b[i]))*result_x
        # u3 = torch.cos((g[i]))*iden - 1j *torch.sin((g[i]))*result_z

        # u1 = u1.to(device)
        # u2 = u2.to(device)
        # u3 = u3.to(device)
        # del result_x 
        # del result_z
        # U = u3 @ u2 @ u1 @ U
        # del u1
        # del u2
        # del u3
        # del iden
        U = torch.matmul((torch.cos((g[i]))*torch.eye(2**N1).to(device) - 1j *torch.sin((g[i]))*result_z),torch.matmul((torch.cos((b[i]))*torch.eye(2**N1).to(device) - 1j *torch.sin((b[i]))*result_x),torch.matmul((torch.cos((a[i]))*torch.eye(2**N1).to(device) - 1j *torch.sin((a[i]))*result_z),U)))
        del result_x
        del result_z
    
    return U

def evolution(time, l, U):
    N1 = N
    L = L1
    global ham

    complex_const = -1j
    ham = (complex_const * time)*ham
    # pdb.set_trace()
    # if ham.is_cuda:
    #     ham = ham.cpu()
    # ham_np = ham.numpy()
    # time = 0.01
    H = ham.matrix_exp()
    H = H.to(device)
    # H = get_H(time)

    U_final = torch.eye(2**N1, dtype=torch.complex64).to(device)
    # U_final = U_final.to(device)
    for l in range(L):
        U_final = torch.matmul(torch.matmul(torch.matmul(U[2*l+1], H), U[2*l]), U_final)
    
    return U_final

def expectation(state,time,U1):
    # pdb.set_trace()
    L = L1
    # state = state.reshape(4, 4)
    # state = torch.flatten(torch.kron(state, torch.tensor([1.,0.])))
    state = state.to(device)
    state = (torch.kron(state, torch.tensor([1.,0.]).to(device)))
    # for l in range(L):
    # with torch.no_grad():
    U = evolution(time, L, U1)
    state = U @ state.view(-1,state.size()[0])
    state = torch.abs(state) ** 2
    exp0 = torch.sum(state[::2], axis = 0)
    exp1 = torch.sum(state[1::2], axis = 0)
    expt = exp0 - exp1
    del exp0
    del exp1
    return expt


#####################  Import Data loader ###########################

from z2states import Z2StateDataset, Z2DatasetLoader, Z3StateDataset, Z3DatasetLoader, DatasetLoader

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pdb

import torch.autograd.profiler as profiler

import torch.nn as nn
import torch.optim as optim

GLOBAL_LIST = []


num_layers = 0

class QuantumPerceptron(nn.Module):
    """"
    mfaPerceptron
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(QuantumPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)
        # self.mid_layers = nn.ModuleList()
        # for _ in range(num_layers):
        #     self.mid_layers.append(nn.Linear(hidden_size, hidden_size))
        # self.layer2 = nn.Linear(hidden_size, output_size)
        # self.layer1 = nn.Linear(input_size, output_size)
        L = L1
        N1 = N
        a = torch.normal(mean=0.0, std=1., size=[L,2,N1]).to(device)
        b = torch.normal(mean=0.0, std=1., size=[L,2,N1]).to(device)
        g = torch.normal(mean=0.0, std=1., size=[L,2,N1]).to(device)
        self.params = nn.Parameter((torch.stack((a,b,g))))
        self.params.requires_grad = True

        # self.params = nn.Parameter(torch.tensor([1.,0.,0.,0.,0.,0.,0.,0]))

    def init_r(self, state, U):
        self.r = []
        for t in torch.arange(0.1,1.,0.1):
            expectations = expectation(state, t, U)
            self.r.append(expectations)
        # self.r.append(torch.tensor(1.).to(torch.float32))
        self.r = torch.stack(self.r)
        return self.r
    
    # def evolve(self, state):
    #     self.r = []
    #     for t in torch.arange(0.01,0.1,0.02):
    #         state_final, expectations = expectation(state, t, self.params)
    #         # self.r.append(state_final)
    #         self.r = state_final
    #     # self.r.append(torch.tensor(1.).to(torch.float32))
    #     # self.r = torch.stack(self.r)
    #     return self.r

    def forward(self, state, U):
        final_state = self.init_r(state, U)
        self.r = final_state.to(torch.float32)
        self.r = self.r.T
        # pdb.set_trace()
        # print(self.params)
        # self.r = state.to(torch.float32)
        # self.r = torch.tensor(r).to(torch.float32)
        # state = torch.tensor(state).reshape(1,16)
        # self.r = state.to(torch.float32)
        # print(self.r)
        out = self.layer1(self.r)
        # out = F.gelu(out)
        # for layer in self.mid_layers:
        #     out = layer(out)
        #     out = F.gelu(out)
        # out = self.layer2(out)
        out = F.relu(out)
        # return return_energy(out)
        return out
        # return out


model = QuantumPerceptron(input_size= 9, output_size= 1, hidden_size = 1).to(device)
# model = QuantumPerceptron(input_size= 9, output_size= 1, hidden_size = 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
model = model
# params = torch.randn(8)  # initial 
# Initialize parameters with Gaussian distribution centered at 0
# import matplotlib.pyplot as plt
# Set a seed for reproducibility
torch.manual_seed(100)

x = Z2DatasetLoader()
y = Z3DatasetLoader()
z = DatasetLoader()

z2_dataset, z2_dataloader = x.return_dataset()
z3_dataset, z3_dataloader = y.return_dataset()

dataset, dataloader = z.return_dataset()

# import pandas as pd

# Assume we have a DataLoader named 'dataloader'

# Create lists for each column in the DataFrame
Js = []
eigenvalues = []
eigenvectors = []
vals = []


for epoch in range(2000):
    print("epoch:", epoch)
    try:
        losses = []
        total_loss = torch.tensor(0.0)
        # a = model.params[0]
        # b = model.params[1]
        # g = model.params[2]
        # L = L1
        # U = []
        # for l in range(L):
        #     U1 = get_U(a[l][0],b[l][0],g[l][0]).to(device)
        #     U2 = get_U(a[l][1],b[l][1],g[l][1]).to(device)
        #     U.append(U1)
        #     U.append(U2)
        model.train()
        # total_loss = torch.tensor(0.0)

        for state, val in tqdm(dataloader):

            a = model.params[0]
            b = model.params[1]
            g = model.params[2]
            L = L1
            U = []
            for l in range(L):
                U1 = get_U(a[l][0],b[l][0],g[l][0]).to(device)
                U2 = get_U(a[l][1],b[l][1],g[l][1]).to(device)
                U.append(U1)
                U.append(U2)

            state = state.to(device)
            val = val.to(device)
            pred = model(state, U)
            pred = pred.reshape(-1,)
            loss = criterion(pred, val.float()).to(device)
            # loss = loss.to(device)
            total_loss = total_loss.to(device)
            total_loss += loss

            # print(pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            

        losses.append(total_loss.item())
        # total_loss.backward()
        # optimizer.step()  
        # optimizer.zero_grad()          
        print("Backward prop...")
        print(total_loss)


        # for param in model.parameters():
        #     param.data.clamp_(0, 2*torch.pi)

        # Clip the parameters to be within the bounds
        model.eval()

    # We don't need to track gradients for validation, so wrap in 
    # no_grad to save memory

        if epoch % 5 == 0:
            print("Starting accuracy calculation...")
            with torch.no_grad():

                correct_predictions = 0
                total_predictions = 0

                print("Iterating through all (for accuracy)...")    

                for state, val in tqdm(dataloader):
                    state = state.to(device)
                    val = val.to(device)
                    pred = model(state, U)

                    # Convert predictions and true values to -1 or 1
                    pred_rounded = torch.where(pred < 0, -1, 1).reshape(-1,)
                    val_rounded = torch.where(val.float() > 0., 1, -1).reshape(-1,)

                    # Count correct predictions
                    correct_predictions += torch.sum(pred_rounded == val_rounded).item()

                    total_predictions += len(val)

                accuracy = correct_predictions / total_predictions
                print(f'Accuracy: {accuracy}')
    
    except KeyboardInterrupt:
        pdb.set_trace()
        continue

# pdb.set_trace()
file_path = './quantumPercPhase2.pth'
torch.save(model.state_dict(),file_path)