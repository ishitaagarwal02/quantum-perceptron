import torch
from torch.utils.data import Dataset, DataLoader
import random
# from heisenberg import Quantumdataset, QuantumDataset
from tqdm import tqdm
from functools import reduce
from scipy.sparse.linalg import expm_multiply
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn 
from z2states import Z2StateDataset, Z2DatasetLoader, Z3StateDataset, Z3DatasetLoader

x = Z2DatasetLoader()
y = Z3DatasetLoader()

torch.manual_seed(42)
complex_const = -1j



def get_H(time):
        
    time = time
    j = [1,1,1,1]
    omega = -20
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

    for j in range(N1-1):
        matrices = []
        for i in range(N1-1):
            m=n if i==j else p
            matrices.append(m)
        matrices.append(n)    
        result = reduce(torch.kron, matrices)
        matrix3 += inter[j]*result

    complex_const = -1j
    ham = (complex_const * time)*(matrix - matrix2 + matrix3)
    # pdb.set_trace()
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
        U = u3 @ u2 @ u1 @ U

    return U

def evolution(time, params,l):
    N1 = 5
    L = 4
    a = params[0]
    b = params[1]
    g = params[2]
    H = get_H(time)

    U_final = torch.eye(2**N1, dtype=torch.complex128)
    U1 = get_U(a[l][0],b[l][0],g[l][0])
    U2 = get_U(a[l][1],b[l][1],g[l][1])
    # pdb.set_trace()
    U_result = torch.matmul(torch.matmul(U2, H), U1)
    U_final = torch.matmul(U_result, U_final)
    # for l in range(L):
    #     U1 = get_U(a[l][0],b[l][0],g[l][0])
    #     U2 = get_U(a[l][1],b[l][1],g[l][1])
    #     # pdb.set_trace()
    #     U_result = U2 * H * U1
    #     U_final = U_result * U_final
    
    return U_final

def expectation(state,time,params):
    N1 = 5
    # pdb.set_trace()
    L=4
    state = state.reshape(4, 4)
    state = torch.flatten(torch.kron(state, torch.tensor([1.,0.])))
    for l in range(L):
        U = evolution(time,params,l)
        state = U @ state
    # state_final = U @ state
    # norm = torch.norm(state_final)
    # state_final = state_final/norm
    # state_final = torch.abs(state_final)**2
    # state_final = torch.abs(torch.dot(U, state)) ** 2
    state_final = torch.abs(state) ** 2
    exp0 = torch.sum(state_final[::2])
    exp1 = torch.sum(state_final[1::2])
    expectation = exp0 - exp1
    return state_final, expectation


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
        # self.layer2 = nn.Linear(input_size, output_size)
        L = 4
        N1 = 5
        a = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        b = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        g = torch.normal(mean=0.0, std=1., size=[L,2,N1])
        self.params = nn.Parameter(torch.zeros_like(torch.stack((a,b,g))))
        self.params.requires_grad = True

        # self.params = nn.Parameter(torch.tensor([1.,0.,0.,0.,0.,0.,0.,0]))

    def init_r(self, state):
        self.r = []
        for t in torch.arange(0.1,1.,0.1):
            state_final, expectations = expectation(state, t, self.params)
            self.r.append(expectations)
        # self.r.append(torch.tensor(1.).to(torch.float32))
        self.r = torch.stack(self.r)
        return self.r
    
    # def evolve(self, state):
    #     self.r = []
    #     for t in torch.arange(0.01,0.1,0.01):
    #         state_final, expectations = expectation(state, t, self.params)
    #         # self.r.append(state_final)
    #         self.r = state_final
    #     self.r.append(torch.tensor(1.).to(torch.float32))
    #     # self.r = torch.stack(self.r)
    #     return self.r

    def forward(self, state):
        final_state = self.init_r(state)
        self.r = final_state.to(torch.float32)
        self.r = self.r.reshape(1, -1)
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
        out = F.tanh(out)
        # return return_energy(out)
        return out
        # return out



z2_dataset, z2_dataloader = x.return_dataset()
z3_dataset, z3_dataloader = y.return_dataset()

model = QuantumPerceptron(input_size= 9, output_size= 1, hidden_size = 144)
model.load_state_dict(torch.load('./quantumPercPhase3.pth'))
# model.eval()
predictions_g = []
predictions_mfa = []

print("Starting accuracy calculation...")
with torch.no_grad():

    correct_predictions = 0
    total_predictions = 0

    print("Iterating through XY (for accuracy)...")    
    for state, val in tqdm(z2_dataloader):
        pred = model(state)
        predictions_g.append(pred.item())
        print("state:", state)
        print("val:", val)
        print("prediction:", pred)
        # Convert predictions and true values to -1 or 1
        pred_rounded = torch.where(pred < 0, -1, 1)
        val_rounded = torch.where(val > 0., 1, -1)
        # pdb.set_trace()

        # Count correct predictions
        correct_predictions += torch.sum(pred_rounded == val_rounded).item()

        total_predictions += len(val)

    print("Iterating through XY_MFA (for accuracy)...")
    for state, val in tqdm(z3_dataloader):
        pred = model(state)
        predictions_mfa.append(pred.item())

        print("state:", state)
        print("val:", val)
        print("prediction:", pred)
        # Convert predictions and true values to -1 or 1
        pred_rounded = torch.where(pred < 0, -1, 1)
        val_rounded = torch.where(val > 0., 1, -1)

        # Count correct predictions
        correct_predictions += torch.sum(pred_rounded == val_rounded).item()

        total_predictions += len(val)

    accuracy = correct_predictions / total_predictions
    print(f'Accuracy: {accuracy}')

# import matplotlib.pyplot as plt
# print("_____predictions of entangled state_____")
# print(predictions_g)
# print("_____predictions of separable state_____")
# print(predictions_mfa)
# plt.plot(predictions_g)
# plt.show()