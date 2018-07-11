import pandas as pd

import torch
import numpy as np
from PIL import Image 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model   
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    
    img = Image.open(image) 
    img.thumbnail((256,256))
    awidth, aheight = img.size
    bwidth, bheight = 224, 224
    l = (awidth - bwidth)/2
    t = (aheight - bheight)/2
    r = (awidth + bwidth)/2
    b = (aheight + bheight)/2
    img = img.crop((l, t, r, b))
    
    np_image = np.array(img)
    np_image = np_image / 255
    np_image = (np_image - mean)/ std
    np_image = np_image.transpose((2,0,1))
    
    return np_image
