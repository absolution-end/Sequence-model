import torch
import random

import io
import os
import unicodedata
import string 
import glob

# alphabets small + capital letters + " .,;"
ALL_Letters = string.ascii_letters  + " .,;"