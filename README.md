## Clone DGAZE repository
- git clone 
- cd DGAZE repo 

## Download DGAZE dataset from the provided link in DGAZE repository
- It will download a folder names "dataset_download"(20GB approx), put it in DGAZE repository

## Code to get drop rate by aligning driver view, projected road view and actual road video (Reference Code)
- We already provide drop rate for each sample corresponding to each driver so this step can be skipped, providing code for reference.
- python codes/drop_rate.py --path "Path to DGAZE folder"

## Run following commmand to get dataset folder 
- python codes/dataset.py --path "Path to DGAZE folder" 

## Visualize the DGAZE dataset
- Visualize.ipynb file contains driver view and road view with gaze point variation across whole sample video for each driver.

## deep-features-video
- Script to extract CNN features from video frames using a Keras pre-trained VGG-19 model.
ii
