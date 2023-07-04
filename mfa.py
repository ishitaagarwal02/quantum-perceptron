import torch
from functools import reduce
import pdb
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

torch.manual_seed(42)

pi = 3.1415927410125732
theta = torch.tensor([pi/2,pi,pi/3,pi/6])
phi = torch.tensor([pi,2*pi,3*pi,pi])

def return_energy(params):
    theta = torch.tensor(params[:4])
    phi = torch.tensor(params[4:])
    N1 = 4
    matrices = ([])
    for i in torch.arange(N1):
        v = torch.tensor([torch.cos(theta[i]), torch.exp(1j*phi[i])*torch.sin(theta[i])])
        matrices.append(v)
        result1 = torch.flatten(torch.tensor(reduce(torch.kron, matrices)))

    matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
    ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

    p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    J = -1

    for j in range(N1-1):
        matrices = []
        for i in range(N1):
            m = s1 if i == j or i==(j+1) else p
            matrices.append(m)
        # pdb.set_trace()
        result = reduce(torch.kron, matrices)
        matrix += result

    for j in range(N1-1):
        matrices = []
        for i in range(N1):
            m = s2 if i == j or i==j+1 else p
            matrices.append(m)
        # print(matrices)
        # pdb.set_trace()
        result = reduce(torch.kron, matrices)
        matrix2 += result

    for j in range(N1-1):
        matrices = []
        for i in range(N1):
            m=s3 if i==j or i==j+1 else p
            matrices.append(m)
        result = reduce(torch.kron, matrices)
        # pdb.set_trace()
        matrix3 += result

    ham = J*(matrix + matrix2 + matrix3)
    # print(ham)
    result_conj_transpose = torch.conj(result1).T
    # pdb.set_trace()
    result1 = result1.unsqueeze(-1).to(torch.complex128)
    dummy =  torch.mm(ham, result1)
    result_conj_transpose = result_conj_transpose.unsqueeze(0).to(torch.complex128)
    energy = torch.mm(result_conj_transpose, dummy)
    #`pdb.set_trace()
    return energy.float()


def print_status(params):
    f = return_energy(params)
    print(f)

# # This is the function we want to minimize
# from scipy.optimize import minimize
# # These are some other parameters of the function
# # This is our initial guess
# params = [pi/2,pi,pi/3,pi/6, pi,2*pi,3*pi,pi]

# from scipy.optimize import Bounds

# # Define the bounds
# bounds = Bounds([0]*8, [2*pi]*8)

# params = [pi/2,pi,pi/3,pi/6, pi,2*pi,3*pi,pi]
# params = [pi,0,0,0,0,0,0,0]
# # params = [pi/8**0.5]*8


# result = minimize(return_energy, params, callback= print_status, method='SLSQP', bounds=bounds)

# print(result)

# import torch.nn as nn
# import torch.optim as optim
# class mfaPerceptron(nn.Module):
#     """"
#     mfaPerceptron
#     """
#     def __init__(self, input_size, hidden_size, output_size):
#         super(mfaPerceptron, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.layer2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out = self.layer1(x).relu()
#         out = self.layer2(x)
#         return out
    
# model = mfaPerceptron(1, 4, 1)

# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# for epoch in range(1000):
#     outputs = model()
#     loss = criterion(outputs, torch.tensor(-3.0))

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print("{} || {}".format(loss, outputs))


import torch.nn as nn
import torch.optim as optim

class mfaPerceptron(nn.Module):
    """"
    mfaPerceptron
    """        
    def __init__(self, input_size, hidden_size, output_size, J):
        super(mfaPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.mid_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.45)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.J = J
        self.J.requires_grad = False

        # self.params = nn.Parameter(torch.tensor([1.,0.,0.,0.,0.,0.,0.,0]))

    def return_energy(self, params):
        theta = (params[:4])
        phi = (params[4:])
        N1 = 4
        matrices = ([])
        for i in torch.arange(N1):
            v = torch.tensor([torch.cos(theta[i]), torch.exp(1j*phi[i])*torch.sin(theta[i])], requires_grad = True)
            matrices.append(v)
            result1 = torch.flatten((reduce(torch.kron, matrices)))

        matrix = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
        matrix2 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
        matrix3 = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)
        ham = torch.zeros((2**N1, 2**N1), dtype=torch.complex128)

        p = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
        s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

        for j in range(N1-1):
            matrices = []
            for i in range(N1):
                m = s1 if i == j or i==(j+1) else p
                matrices.append(m)
            # pdb.set_trace()
            result = reduce(torch.kron, matrices)
            matrix += result

        for j in range(N1-1):
            matrices = []
            for i in range(N1):
                m = s2 if i == j or i==j+1 else p
                matrices.append(m)
            # print(matrices)
            # pdb.set_trace()
            result = reduce(torch.kron, matrices)
            matrix2 += result

        for j in range(N1-1):
            matrices = []
            for i in range(N1):
                m=s3 if i==j or i==j+1 else p
                matrices.append(m)
            result = reduce(torch.kron, matrices)
            # pdb.set_trace()
            matrix3 += result

        ham = self.J*(matrix + matrix2 + matrix3)
        # print(ham)
        result_conj_transpose = torch.conj(result1).T
        #`pdb.set_trace()
        # pdb.set_trace()
        result1 = result1.unsqueeze(-1).to(torch.complex128)
        dummy =  torch.mm(ham, result1)
        result_conj_transpose = result_conj_transpose.unsqueeze(0).to(torch.complex128)
        energy = torch.mm(result_conj_transpose, dummy)
        #`pdb.set_trace()
        return energy.float(), result1

    def forward(self, x):
        out = self.layer1(x)
        out = self.mid_layer(out)
        out = self.dropout(out)
        out = self.layer2(out)
        # return return_energy(out)
        mf_energy, result1 = self.return_energy(out)
        return mf_energy, out, result1
        # return out

input_size = 8  # number of parameters in the return_energy function
output_size = 8  # since return_energy returns a single value


J_values = []
output_energy_list = []
output_vectors_list = []
output_states_list = []
labels_list = []


for J in tqdm(torch.arange(-1, 1.01, 0.1)):
        
    params = torch.normal(mean=0, std=1, size=(8,))

    x = torch.ones([8,]) + torch.randn(8)

    model = mfaPerceptron(input_size= input_size, output_size= output_size, hidden_size = 16, J = J)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    # params = torch.randn(8)  # initial 
    # Initialize parameters with Gaussian distribution centered at 0

    # Set a seed for reproducibility
    torch.manual_seed(0)

    good_results_outputs = ([])
    good_results_outs = ([])
    good_results_result1 = ([])


    for epoch in range(500):
        params = params.requires_grad_()  # make sure gradients are computed with respect to params
        #`pdb.set_trace()
        outputs, out, result1 = model(x)
        # target = torch.tensor([-3.0])  # target energy is -3
        # loss = criterion(outputs, target)
        # print(loss)
        good_results_outputs.append(outputs)
        good_results_outs.append(out)
        good_results_result1.append(result1)

        optimizer.zero_grad()
        outputs.backward()
        optimizer.step()
        # Clip the parameters to be within the bounds
        for param in model.parameters():
            param.data.clamp_(0, 2 * pi)

        # if epoch % 25 == 0:
        #     print("{} || {} || {}/500".format(out, outputs, epoch))
        #     print("Parameters: ")
            # for name, param in model.named_parameters():
            #     print("{}: {}".format(name, param.data))
    try:
        idx = torch.argmin(torch.tensor(good_results_outputs))
        J_values.append(J)
        output_energy_list.append(good_results_outputs[idx])
        output_vectors_list.append(good_results_outs[idx])
        output_states_list.append(good_results_result1[idx])
        labels_list.append(1)
    except:
        pdb.set_trace()


print("!!!!!!!!!!!!!!!!!! PREPROCESSING OVER !!!!!!!!!!!!!!!!!!!!!!!!!")


from torch.utils.data import Dataset, DataLoader

class MFADataset(Dataset):
    def __init__(self, J_values, output_energy_list, output_vectors_list, output_state_list, labels_list):
        self.J_values = J_values
        self.output_energy_list = output_energy_list
        self.output_vectors_list = output_vectors_list
        self.output_state_list = output_state_list

    def __len__(self):
        return len(self.J_values)

    def __getitem__(self, idx):
        J = self.J_values[idx]
        output_energy = self.output_energy_list[idx]
        output_vectors = self.output_vectors_list[idx]
        output_states = self.output_state_list[idx]
        label = labels_list[idx]
        return J, output_energy, output_vectors, label

# Assuming you have the necessary variables J_values, good_results_outputs, and good_results_outs

# Create a PyTorch DataLoader
dataset = MFADataset(J_values= J_values,  output_energy_list= output_energy_list, output_vectors_list = output_vectors_list, labels_list= labels_list, output_state_list= output_states_list)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

class MFADatasetLoader():
    def return_dataset(self):
        dataset = MFADataset(J_values= J_values,  output_energy_list= output_energy_list, output_vectors_list = output_vectors_list, labels_list= labels_list, output_state_list= output_states_list)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataset, dataloader

# Iterate over the dataloader
for J, good_results_output, good_results_out, value in dataloader:
    print(f"J: {J.item()}")
    print(f"Good Results Output: {good_results_output.item()}")
    print(f"Good Results Out: {good_results_out}")
    print(f"Value: {value.item()}")
    print("--------------------------")
    
