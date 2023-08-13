import torch
from functools import reduce
# from scipy.linalg import expm
# import scipy.sparse as sparse
# import scipy.linalg
# from scipy.sparse.linalg import expm_multiply
import numpy as np
from torch.utils.data import Dataset, DataLoader

D = 9
# torch.manual_seed(0)
def z1phase():
    
    N1 = D
    states00 = []
    states11 = []
    zero_state = torch.tensor([1, 0], dtype=torch.complex64)
    one_state = torch.tensor([0, 1], dtype=torch.complex64)
    matrix = []
    # for i in torch.arange(15):
        # generate 0000 type states
    for j in torch.arange(N1):
        r = torch.rand(1) * (1. - 0.7) + 0.7
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # = torch.cos(phi) + 1j*torch.sin(phi)
        # = torch.tensor(]) 
        s = torch.sqrt(r)*zero_state + torch.sqrt(1-r)*one_state
        matrix.append(s)
    
    states11 = reduce(torch.kron, matrix)
    matrix = []
    # generate 1111 type states
    for j in torch.arange(N1):
        r = torch.rand(1) * (1. - 0.7) + 0.7
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # = torch.cos(phi) + 1j*torch.sin(phi)
        # = torch.tensor(]) 
        s = torch.sqrt(1-r)*zero_state + torch.sqrt(r)*one_state
        # pdb.set_trace()
        matrix.append(s)
        
    states00 = reduce(torch.kron, matrix) 
    return states11, states00


def z2phase():
    
    N1 = D
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

def z3phase():
    N1 = D
    states100 = []
    states010 = []
    states001 = []
    zero_state = torch.tensor([1, 0], dtype=torch.complex64)
    one_state = torch.tensor([0, 1], dtype=torch.complex64)
    matrix = []
    # for i in torch.arange(15):
    # generate 001 type states
    for j in torch.arange(N1):
        r = 0.3 * torch.rand(1)
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # e_phi = torch.cos(phi) + 1j*torch.sin(phi)
        # e_phi = torch.tensor(e_phi) 
        # print(r)
        if j%3==0:
            s = torch.sqrt(r)*zero_state + torch.sqrt(1-r)*one_state
        else:
            s = torch.sqrt(1-r)*zero_state + torch.sqrt(r)*one_state
        matrix.append(s)
    
    states100 = reduce(torch.kron, matrix)
    # print(states100)
    matrix = []
    # generate 010 type states
    for j in torch.arange(N1):
        r = 0.3 * torch.rand(1)
        # print(r)
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # e_phi = torch.cos(phi) + 1j*torch.sin(phi)
        # e_phi = torch.tensor(e_phi) 
        if j%3==1:
            s = torch.sqrt(r)*zero_state + torch.sqrt(1-r)*one_state
        else:
            s = torch.sqrt(1-r)*zero_state + torch.sqrt(r)*one_state
        # pdb.set_trace()
        matrix.append(s)

    states010 = reduce(torch.kron, matrix)  
    # print(states010)  
    matrix = []
    # generate 001 type states
    for j in torch.arange(N1):
        r = 0.3 * torch.rand(1)
        r = torch.tensor([r]) 
        # phi = 2 * torch.pi * torch.rand(1)
        # e_phi = torch.cos(phi) + 1j*torch.sin(phi)
        # e_phi = torch.tensor(e_phi) 
        if j%3==2:
            s = torch.sqrt(r)*zero_state + torch.sqrt(1-r)*one_state
        else:
            s = torch.sqrt(1-r)*zero_state + torch.sqrt(r)*one_state
        # pdb.set_trace()
        matrix.append(s)
        
    states001 = reduce(torch.kron, matrix)   
    # print(states001)

    return states100, states010, states001


z2state_list = []
z2label_list = []
for i in torch.arange(18):
    s21, s22 = z2phase()
    z2state_list.append(s21)
    z2state_list.append(s22)
    z2label_list.append(torch.tensor(-1.))  # Adding label -1 for each state
    z2label_list.append(torch.tensor(-1.))

# print(tracemalloc.get_traced_memory())
class Z2StateDataset(Dataset):
    def __init__(self, z2state_list, z2label_list):
        self.state_list = z2state_list
        self.label_list = z2label_list

    def __len__(self):
        return len(self.state_list)

    def __getitem__(self, idx):
        return self.state_list[idx], self.label_list[idx]

dataset_z2 = Z2StateDataset(z2state_list, z2label_list)

# Create the dataloader
dataloader_z2 = DataLoader(dataset_z2, batch_size=18, shuffle=False)

class Z2DatasetLoader():
    def return_dataset(self):
        dataset = Z2StateDataset(z2state_list, z2label_list)
        dataloader = DataLoader(dataset, batch_size=18, shuffle=False)
        return dataset, dataloader
    
for states, labels in dataloader_z2:
    # pdb.set_trace()
    print(states, labels)

# print(tracemalloc.get_traced_memory())


z3state_list = []
z3label_list = []
for i in torch.arange(12):
    s31, s32, s33 = z3phase()
    z3state_list.append(s31)
    z3state_list.append(s32)
    z3state_list.append(s33)
    z3label_list.append(torch.tensor(1.))  # Adding label -1 for each state
    z3label_list.append(torch.tensor(1.))
    z3label_list.append(torch.tensor(1.))

class Z3StateDataset(Dataset):
    def __init__(self, z3state_list, z3label_list):
        self.state_list = z3state_list
        self.label_list = z3label_list

    def __len__(self):
        return len(self.state_list)

    def __getitem__(self, idx):
        return self.state_list[idx], self.label_list[idx]

dataset_z3 = Z3StateDataset(z3state_list, z3label_list)

class Z3DatasetLoader():
    def return_dataset(self):
        dataset = Z3StateDataset(z3state_list, z3label_list)
        dataloader = DataLoader(dataset, batch_size=18, shuffle=True)
        return dataset, dataloader

# Create the dataloader
dataloader_z3 = DataLoader(dataset_z3, batch_size=18, shuffle=True)
for states, labels in dataloader_z3:
    print(states, labels)

# print(tracemalloc.get_traced_memory())

from torch.utils.data import ConcatDataset

dataset_combined = ConcatDataset([dataset_z2, dataset_z3])
dataloader_combined = DataLoader(dataset_combined, batch_size=9, shuffle=True)

class DatasetLoader():
    def return_dataset(self):
        return dataset_combined, dataloader_combined
# print(tracemalloc.get_traced_memory())

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')

# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)
    
# import os


# print("memory time")
# def get_memory_info():
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024**2  # in MB

# print(get_memory_info())
# tracemalloc.stop()
