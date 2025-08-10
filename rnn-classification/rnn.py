import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from utils import ALL_Letters , N_Letters
from utils import load_data , letter_to_index, letter_to_tensor, random_training_eg


class RNN(nn.Module):
    '''Implementing RNN from scratch'''
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i20 = nn.Linear()