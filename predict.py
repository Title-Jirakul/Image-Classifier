# PROGRAMMER: Title J.
# DATE CREATED: 07/10/2018         
# PURPOSE: Predict flower name from an image along with the probability of that name.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --dir <directory with an image> --topK <number of topK>
#             --device <device used> --save_dir <directory to checkpoint> 
#             --json_dir <directory to json file>
#   Example call:
#    python train.py --dir flowers/test/1/flower1.png --topK 5  
#             --device cuda --save_dir checkpoint.pth
#             --json_dir cat_to_names.json
##  

# Imports python modules
import argparse
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import numpy as np
import torch.nn.functional as F
import json
import checkpoint_loader as cloader
import processing_images as pi

# Main program function defined below
def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    print("Command Line Arguments:\n   dir = ", in_arg.dir, "\n   topK = ", in_arg.topK,
          "\n   device = ", in_arg.device, "\n   save_dir = ", in_arg.save_dir, 
          "\n   json_dir = ", in_arg.json_dir, "\n")

    # Load in a mapping from image category label to category name
    with open(in_arg.json_dir, 'r') as f:
        cat_to_name = json.load(f)
    
    # Load model from the checkpoint
    model, class_to_idx = cloader.load_checkpoint(in_arg.save_dir)
    
    # Obtain the topK probabilities and classes of the image
    probs, classes = predict(in_arg.dir, model, in_arg.topK, in_arg.device, class_to_idx)
    
    # print out prediction
    str_classes = np.char.mod('%d', classes)
    names = list()
    for n in range(5):
        names.append(cat_to_name.get(str_classes[n]))
    flower_names = np.array(names)
    
    print(flower_names)
    print(probs)


# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 5 command line arguments args.dir for path to images file,
    # args.topK for top K classes , args.device for training the 
    # model using GPU or CPU, , args.save_dir path to directory to save checkpoints,
    # args.json_dir for directory to JSON file
    parser.add_argument('--dir', type=str, default='flowers/train/1/image_06738.jpg', 
                        help='path to image for prediction')
    parser.add_argument('--topK', type=int, default=5,
                        help='number of topK')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='choosing between CPU and GPU to train the model')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', 
                        help='directory to save checkpoints')
    parser.add_argument('--json_dir', type=str, default='cat_to_name.json', 
                        help='directory to JSON file that maps the class values to other category names')

    # returns parsed argument collection
    return parser.parse_args()

def predict(image_path, model, topk, device, class_to_idx):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    a = pi.process_image(image_path) 
    y = np.expand_dims(a, axis=0) 
    model.to(device)
    model.eval()
    if (device == 'cuda'):
        img = torch.from_numpy(y).float().cuda()
    else:
        img = torch.from_numpy(y).float()

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)
    top = ps.topk(topk)
    
    probs = top[0].cpu().numpy()[0]
    index_classes = top[1].cpu().numpy()[0]
    
    classes = np.full(5, 0)
    count = 0
    for search_index in index_classes:
        for label, index in class_to_idx.items():  
            if index == search_index:
                classes[count] = label
        count += 1
        
    return probs, classes


# Call to main function to run the program
if __name__ == "__main__":
    main()