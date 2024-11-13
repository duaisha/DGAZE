import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import math
import cv2
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms as T

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:0')
torch.manual_seed(17)

import math
import copy
import os, sys



# Function to log messages to both the console and a file
def log_message(message, logfile='training_log.txt'):
    print(message)
    with open(logfile, 'a') as f:
        f.write(message + '\n')

# Testing function
def test(model, testdataset, batch_size=64, logfile='training_log.txt'):
    criterion = nn.L1Loss(reduction='sum')
    loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss = 0
    numsamples = 0
    with torch.no_grad():
        for ii, (leye, x, gt) in enumerate(loader):
            out = model(leye.to(device), x.to(device))
            batch_loss = criterion(out, gt.to(device)).item()
            loss += batch_loss
            numsamples += len(gt)
            
#             log_message(f'Testing: Processed {numsamples}/{len(testdataset)} samples', logfile)

    avg_loss = loss / numsamples
#     avg_loss = loss 
    log_message(f'Testing completed. Average loss: {avg_loss:.4f}\n', logfile)
    return avg_loss

# Training function
def train(model, traindataset, testdataset, batch_size=32, nepochs=10, dump_path=None, logfile='training_log.txt'):
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=0)

    logfile = os.path.join(dump_path, logfile)
    log_message('Training started\n', logfile)

    for epoch in range(nepochs):
        running_loss = 0
        numsamples = 0
        log_message(f'Starting Epoch {epoch + 1}/{nepochs}', logfile)

        # Training loop
        for ii, (leye, feature, gt) in enumerate(train_loader):
            out = model(leye.to(device), feature.to(device))
            loss = criterion(out, gt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update tracking metrics
            batch_loss = loss.item()
            running_loss += batch_loss
            numsamples += gt.shape[0]

            if (ii + 1) % 10 == 0:
#                 avg_batch_loss = running_loss 
#                 avg_batch_loss = running_loss / numsamples
#                 log_message(f'Epoch [{epoch + 1}/{nepochs}], Batch [{ii + 1}/{len(train_loader)}], '
#                             f'Average Loss: {avg_batch_loss:.4f}', logfile)
                running_loss = 0
                numsamples = 0

        # Epoch summary
        log_message(f'Completed Epoch {epoch + 1}/{nepochs}', logfile)
        train_error = test(model, traindataset, batch_size, logfile)
        test_error = test(model, testdataset, batch_size, logfile)
        log_message(f'\tTrain Error: {train_error:.4f}\n\tTest Error: {test_error:.4f}\n', logfile)

        # Save model after each epoch
        torch.save(model.state_dict(), f'Model_epoch_{epoch + 1}.pth')
        log_message(f'Model saved after Epoch {epoch + 1}\n', logfile)