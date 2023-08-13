import torch
import pdb
from functools import reduce
# from scipy.linalg import expm
# import scipy.sparse as sparse
# import scipy.linalg
import numpy as np
# from scipy.sparse.linalg import expm_multiply
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import tracemalloc
import psutil

D = 8
tracemalloc.start()
# torch.manual_seed(0)

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

z2state_list = []
z2label_list = []
for i in torch.arange(36):
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
    
import os


print("memory time")
def get_memory_info():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB

print(get_memory_info())