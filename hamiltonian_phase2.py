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
import psutil
from memory_profiler import profile


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2
# pip import memory-profiler

complex_const = -1j

N = 7
L1 = 4

N1 = N
j = [1] * (N1-1)
omega = -20
rabif =  [0] * (N1 - 1) + [omega]
detun = [2 * val for val in j] + [2 * sum(j)]
inter = [4 * val for val in j]


N1 = N
# Initialize the resulting matrix as zero
ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex64)

s = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
n = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64)
print("")

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
# @profile


def get_U(a,b,g):
    
    gc.collect()
    N1 = N
    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64)
    identity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.complex64)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64)

    U = torch.eye(2**N1, dtype=torch.complex64)

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

    return U

# @profile
def evolution(time, params):

    gc.collect()
    global ham
    N1 = N
    L = L1
    # print("L:",L)
    a = params[0]
    b = params[1]
    g = params[2]
    complex_const = -1j
    ham = (complex_const * time)*(ham)
    H = ham.matrix_exp()

    U_final = torch.eye(2**N1, dtype=torch.complex64)
    for l in range(L):        
        U1 = get_U(a[l][0],b[l][0],g[l][0])
        # print("here")
        U2 = get_U(a[l][1],b[l][1],g[l][1])
        U_final = torch.matmul(torch.matmul(torch.matmul(U2, H), U1), U_final)
        del U1
        del U2
        gc.collect()

    return U_final

def expectation(state,time,params):

    state = (torch.kron(state, torch.tensor([1.,0.])))
    with torch.no_grad():
        U = evolution(time,params)
    # pdb.set_trace()
    state = U @ state.view(-1,state.size()[0])
    del U
    state = torch.abs(state) ** 2
    exp0 = torch.sum(state[::2], axis = 0)
    exp1 = torch.sum(state[1::2], axis = 0)
    expt = exp0 - exp1
    return expt


#####################  Import Data loader ###########################

from z2states import Z2StateDataset, Z2DatasetLoader, Z3StateDataset, Z3DatasetLoader, DatasetLoader

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pdb
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

num_layers = 0

class QuantumPerceptron(nn.Module):
    """"
    mfaPerceptron
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(QuantumPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)
        L = L1
        N1 = N
        a = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        b = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        g = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        self.params = nn.Parameter((torch.stack((a,b,g))))
        self.params.requires_grad = True

    def init_r(self, state):
        self.r = []
        for t in torch.arange(0.1,1.,0.2):
            expectations = expectation(state, t, self.params)
            # print(f"Memory usage after calculating expt: {get_memory_usage()} MB")
            self.r.append(expectations)
        # self.r.append(torch.tensor(1.).to(torch.float32))
        self.r = torch.stack(self.r)
        return self.r
    # @profile
    def forward(self, state):
        final_state = self.init_r(state)
        self.r = final_state.to(torch.float32)
        self.r = self.r.T
        out = self.layer1(self.r)
        out = F.tanh(out)
        return out
        # return out


# model = QuantumPerceptron(input_size= 9, output_size= 1, hidden_size = 1).to(device)
model = QuantumPerceptron(input_size= 5, output_size= 1, hidden_size = 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
model = model

torch.manual_seed(0)

x = Z2DatasetLoader()
y = Z3DatasetLoader()
z = DatasetLoader()

z2_dataset, z2_dataloader = x.return_dataset()
z3_dataset, z3_dataloader = y.return_dataset()

dataset, dataloader = z.return_dataset()

for epoch in range(3):
    print("epoch:", epoch)
    try:
        losses = []
    
        model.train()
        total_loss = torch.tensor(0.0)

        for state, val in tqdm(dataloader):
            # state = state.to(device)
            # val = val.to(device)
            optimizer.zero_grad()
            pred = model(state)
            pred = pred.reshape(-1,)
            loss = criterion(pred, val.float())
            total_loss += loss.item()
            # print(pred)
            print(f"Memory usage before back prop: {get_memory_usage()} MB")

            loss.backward()
            # print(f"Memory usage inter: {get_memory_usage()} MB")
            optimizer.step()   
            # pdb.set_trace()         
            print(f"Memory usage after back prop: {get_memory_usage()} MB")


        # total_loss.backward()
        # optimizer.step()            
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
                    # state = state.to(device)
                    # val = val.to(device)
                    pred = model(state)

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
# file_path = './quantumPercPhase2.pth'
# torch.save(model.state_dict(),file_path)
print("memory::")
def get_memory_info():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB

print(get_memory_info())