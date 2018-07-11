# Imports python modules
import pandas as pd

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import build_classifier as bc
import train

# Load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model, input_size = train.build_model(checkpoint['arch'])
    
    # Freeze the parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    p_dropout = checkpoint['dropout']
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
 
    class_to_idx = checkpoint['class_to_idx']
    
    return model, class_to_idx