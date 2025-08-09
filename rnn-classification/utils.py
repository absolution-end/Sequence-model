import torch
import random

import io
import os
import unicodedata
import string 
import glob

# alphabets small + capital letters + " .,;"
ALL_Letters = string.ascii_letters  + " .,;"

# turn a unicode string to a plan ASCII
def unicode_to_ascii(s):
    return " ".join(
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
        return [unicode_to_ascii(lines) for line in lines ]
    
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categorys.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categorys