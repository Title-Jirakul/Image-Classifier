# PROGRAMMER: Title J.
# DATE CREATED: 07/08/2018                                  
# REVISED DATE: 07/09/2018  - added all the functions in the program
# PURPOSE: Create and train a new network on a dataset of images based on user's input parameters. 
#          The training loss, validation loss, and validation accuracy are printed out as a network trains
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <directory with images> --arch <model>
#             --leaning_rate <learning rate 
#             --hidden<number of hidden layers>
#             --epochs <number of training epochs>
#             --device <running with GPU or CPU> 
#             --save_dir <directory to save checkpoints>
#   Example call:
#    python train.py --dir flowers/ --arch vgg --lr 0.01 --hidden 3 
#             --epochs 3 --device cuda --save_dir checkpoint.pth
##

# Imports python modules
import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import build_classifier as bc

# Main program function defined below
def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    print("Command Line Arguments:\n   dir = ", in_arg.dir, "\n   arch = ", in_arg.arch,
          "\n   learning_rate = ", in_arg.learning_rate, "\n   hidden_units = ", 
          in_arg.hidden_units, "\n   epochs = ", in_arg.epochs,
          "\n   device = ", in_arg.device, "\n   save_dir = ", in_arg.save_dir, "\n")
    
    # Load the data 
    dataloaders, validloader, testloader, image_datasets = load_data(in_arg.dir)
        
    # Build a model and input size
    model, input_size = build_model(in_arg.arch)
    
    # Obtain units in each hidden layer
    hidden_layers = build_layers(in_arg.hidden_units)
    
    # Build a classifier
    for param in model.parameters():
        param.requires_grad = False
    p_dropout = 0.2 
    output_size = 102
    classifier = bc.Network(input_size, output_size, hidden_layers, p_dropout)
    model.classifier = classifier
    
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), in_arg.learning_rate)
    device = 'cuda'
    if in_arg.device == 'cpu':
        device = 'cpu'
    epochs = in_arg.epochs
    
    train_model(device, model, dataloaders, validloader, criterion, optimizer, epochs)
    test_network(device, model, testloader)
    
    #Save the checkpoint
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'dropout': p_dropout,
                  'arch': in_arg.arch,
                  'classifier': classifier,
                  'device': in_arg.device,
                  'class_to_idx': image_datasets.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, in_arg.save_dir)

        
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

    # Creates 7 command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.learning_rate for
    # desired learning rate to train the model, args.hidden_units for number of hidden units
    # in the model, args.epochs for number of training epochs, args.device for training the 
    # model using GPU or CPU, , args.save_dir path to directory to save checkpoints
    parser.add_argument('--dir', type=str, default='flowers', 
                        help='path to folder of images for training')
    parser.add_argument('--arch', type=str, default='vgg', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=3,
                        help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=7,
                        help='number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='choosing between CPU and GPU to train the model')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', 
                        help='directory to save checkpoints')

    # returns parsed argument collection
    return parser.parse_args()

def load_data(image_dir):
    """
    Load the images data and provide transformation to what the network is expected
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
     Returns:
     dataloaders, validloader, testloader
    """
    # Creates data directory
    data_dir = image_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return dataloaders, validloader, testloader, image_datasets

def build_model(model_name):
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    model_dict = {'alexnet': [alexnet, 9216] , 'vgg': [vgg16, 25088]}

    model = model_dict[model_name][0]
    input_size = model_dict[model_name][1]
    
    return model, input_size

def build_layers(hidden):
    hidden_layers = list()
    for n in range(hidden):
        while True:
            try: 
                x = int(input('Enter a unit number 1 or greater: ' ))
                if x >= 1:
                    hidden_layers.append(x)
                    break
                else:
                    raise ValueError('Invalid input') 
            except:
                print('INVALID: The input value has to be 1 or greater')  
    return(hidden_layers)

def validation(model, validloader, criterion, device):
    """
    validation pass
    """
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def train_model(device, model, dataloaders, validloader, criterion, optimizer, epochs):
    print_every = 40
    steps = 0
    running_loss = 0
    
    model.to(device)
    
    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(dataloaders):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            # Reset gradient
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
            
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
                running_loss = 0
            
                # Make sure training is back on
                model.train()

def test_network(device, model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    
# Call to main function to run the program
if __name__ == "__main__":
    main()
