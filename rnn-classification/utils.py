import torch
import random

import io
import os
import unicodedata
import string 
import glob

# alphabets small + capital letters + " .,;"
ALL_Letters = string.ascii_letters  + " .,;"
N_Letters   = len(ALL_Letters)

# turn a unicode string to a plan ASCII
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_Letters
    )
    
def load_data():
    ''' Build category_lines directionary, a list of name per language'''
    category_lines = {}
    all_categorys = []
    
    def find_files(path):
        return glob.glob(path)
    
    def read_lines(filename):
        lines = io.open(filename, encoding ='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines ]
    
    for filename in find_files('G:/Sequence-model/rnn-classification/data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categorys.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categorys


# Letter & Word to Tensor Conversion

def letter_to_index(letter):
    # letter-> index
    return ALL_Letters.find(letter)

def letter_to_tensor(letter):
    # letter -> one-hot Tensor
    tensor = torch.zeros(1, N_Letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    # Word -> 3D
    tensor = torch.zeros(len(line),1,N_Letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)]=1
    return tensor


def random_training_eg(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) -1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    
    return category, category_tensor, line, line_tensor

if __name__=='__main__':
    print(ALL_Letters)  
    
    print(unicode_to_ascii('Ślusàrski'))
    
    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])
    
    print(letter_to_tensor('J')) # [1, 57]
    print(line_to_tensor('Jones').size()) # [5, 1, 57]  