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

from dataset import DGazeDataset
from model import IDGAZE
from train_test import train, test

import math
import copy
import os, sys

if __name__ =='__main__':

    ## Path
    DGAZE_extracted_data = '../../DGAZE_extracted_data/DGAZE_extracted_data.pkl'
    DGAZE_data_split = '../../DGAZE_extracted_data/DGAZE_data_split.pkl'
    experiment_name = ""
    dump_path = '../../results/save_models/' + experiment_name

    ## Training Params
    batch_size = 32
    learning_rate = 0.001
    nepochs = 300

    # Load dictionary
    with open(DGAZE_extracted_data, 'rb') as file:
        driver_data = pickle.load(file)

    # Load dictionary
    with open(DGAZE_data_split, 'rb') as file:
        data_split = pickle.load(file)


    train_dataset = DGazeDataset(driver_data, data_split['drivers_train'], data_split['sequence_train'])
    val_dataset = DGazeDataset(driver_data, data_split['drivers_val'], data_split['sequence_val'])
    test_dataset = DGazeDataset(driver_data, data_split['drivers_test'], data_split['sequence_test'])

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    model = IDGAZE().to(device)
    train(model, train_dataset, val_dataset, batch_size=batch_size, nepochs=nepochs, dump_path=dump_path)