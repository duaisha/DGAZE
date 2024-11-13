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


class DGazeDataset(Dataset):
    def __init__(self, driver_data, drivers, sequences, transform = False):
        self.driver_data = driver_data
        self.drivers = drivers
        self.sequences = sequences
        self.left_eye =  np.empty((0, 36, 60, 3))
        self.facial_features = np.empty((0, 14))
        self.gaze_point = np.empty((0, 2))
#         print(sequences)
        
        for ix, driver in enumerate(drivers):
            print("==>", driver)
            data = driver_data[driver]
            for seq in tqdm(sequences):
                if 'seq' + str(seq) in data.keys():
                    data_seq = data['seq' + str(seq)]
                    self.left_eye = np.concatenate((self.left_eye, data_seq['left_eye']), axis=0)
                    seq_facial_features = np.concatenate((data_seq['headpose_pupil'][:,1:], data_seq['face_location']), axis=-1)
                    self.facial_features = np.concatenate((self.facial_features, seq_facial_features), axis=0)
                    self.gaze_point = np.concatenate((self.gaze_point, data_seq['gaze_point'][:,:2]), axis=0)

        print("Data loaded!")
#         self.normalize_eye_image()
        self.normalize_facial_features() 
        self.fix_gaze_point()
        self.gaze_point[:,0][self.gaze_point[:,0]<0] = 0
        self.gaze_point[:,1][self.gaze_point[:,1]<0] = 0
        self.index = np.arange(len(self.gaze_point))
        
        
    def fix_gaze_point(self):
        self.gaze_point[:,1][self.gaze_point[:,1]>=1080]=1080-1
        self.gaze_point[:,0][self.gaze_point[:,0]>=1920]=1920-1


    def normalize_eye_image(self):
        # Calculate mean and standard deviation per channel
        mean = self.left_eye.mean(axis=(0, 1, 2), keepdims=True)
        std = self.left_eye.std(axis=(0, 1, 2), keepdims=True)

#         print("\nMean before normalization:", self.left_eye.mean(axis=(0, 1, 2)))
#         print("Standard deviation before normalization:", self.left_eye.std(axis=(0, 1, 2)))
        
        # Normalize using mean and standard deviation
        self.left_eye = (self.left_eye - mean) / std

#         # Check the result
#         print("\nMean after normalization:", self.left_eye.mean(axis=(0, 1, 2)))
#         print("Standard deviation after normalization:", self.left_eye.std(axis=(0, 1, 2)))
        
    def normalize_facial_features(self):
        # Column-wise mean and standard deviation
        mean = self.facial_features.mean(axis=0)
        std = self.facial_features.std(axis=0)
        
#         # Verify that each column now has mean 0 and variance 1
#         print("\nMean of each column before normalization:", self.facial_features.mean(axis=0))
#         print("Variance of each column beforeafter normalization:", self.facial_features.var(axis=0))

        # Mean-variance normalization (standardization)
        self.facial_features = (self.facial_features - mean) / std

#         # Verify that each column now has mean 0 and variance 1
#         print("\nMean of each column after normalization:", self.facial_features.mean(axis=0))
#         print("Variance of each column after normalization:", self.facial_features.var(axis=0))
        

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        index = self.index[idx]
        left_eye = self.left_eye[index]
        left_eye = np.transpose(left_eye, (2,0,1))
        left_eye = torch.tensor(left_eye, dtype=torch.float32, device=device)
        facial_features = torch.tensor(self.facial_features[index], dtype=torch.float32, device=device)
        gaze_point = torch.tensor(self.gaze_point[index], dtype=torch.float32, device=device)
        return  left_eye, facial_features, gaze_point
