import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Define the Left Eye Convolutional Model
class LeftEyeModel(nn.Module):
    def __init__(self):
        super(LeftEyeModel, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)  # Flatten the tensor for the next layer
        return x

# Define the Face Feature Dense Model
class FaceFeaturesModel(nn.Module):
    def __init__(self):
        super(FaceFeaturesModel, self).__init__()
        self.fc1 = nn.Linear(14, 16)  # Equivalent to Dense(16, activation='relu', input_dim=14)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

# Define the Merged Model
class IDGAZE(nn.Module):
    def __init__(self):
        super(IDGAZE, self).__init__()
        self.left_eye = LeftEyeModel()
        self.face_features = FaceFeaturesModel()
        
        # Dense Layers after Concatenation
        self.fc2 = nn.Linear(4566, 512)  # Adjust 50*7*7 based on left_eye output dimensions
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x1, x2):
        x1 = self.left_eye(x1)
        x2 = self.face_features(x2)
        
        # Concatenate outputs from the two models
        x = torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model and print the summary
model = IDGAZE()
print(model)