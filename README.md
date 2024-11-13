## Clone DGAZE repository
- git clone 
- cd DGAZE repo 

<<<<<<< HEAD

## Prepare Dataset:

#### Download DGAZE dataset from the provided link in DGAZE repository
- It will download a folder names "dataset_download"(20GB approx), put it in DGAZE repository

#### Code to get drop rate by aligning driver view, projected road view and actual road video (Reference Code)
- We already provide drop rate for each sample corresponding to each driver so this step can be skipped, providing code for reference.
- python Codes/Dataset_codes/drop_rate.py --path "Path to DGAZE folder"

#### Run following commmand to get dataset folder 
- python Codes/Dataset_codes/dataset.py --path "Path to DGAZE folder" 
=======
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
>>>>>>> 818ec89c4b164d53427b6f138b75f33338d4adec

#### Visualize the DGAZE dataset
- Codes/Dataset_codes/visualize_dataset.ipynb file contains driver view and road view with gaze point variation across whole sample video for each driver.


## Extracted features:

   (Download the extracted features used by us or generate one of your own using extracting features steps given in next section)
   - Download the extracted features from google driver 'DGAZE_extracted_data.pkl' and 'DGAZE_data_split.pkl'
   - Add the above downloaded pkl files to DGAZE/DGAZE_extracted_data
   - What  'DGAZE_extracted_data.pkl' contains
     -- left_eye image: (nframes x 36 x 60 x 3)
     -- right_eye image: (nframes x 36 x 60 x 3)
     -- face location/face bounding box: (nframes x 4)
     -- headpose_pupil: (nframes x 11)
        - (nframes, roll, pitch, yaw, lpupil(x,y), rpupil(x,y), face_area, nose(x,y))
     -- gaze_point: (nframesx6)
        - First two values are center of object bounding box (x,y)
        - Next four are x,y corresponding to top left point of the object and bottom right of the object respectively.

## SOTA used to Extract Features:

#### Feature Branch
1. Face Detection
   - Using DLIB library 
   - It gives Face Bounding Box and Facial Landmarks including pupil location 
   - We use Face bounding box and nose to get the location of face in the scene
   
2. Area of face bounding box
   - Use abount face bounding box to get the area 
   - It helps us to distance of the head from the screen assuming that facial area does not vary much between drivers.
  
3. Head Pose
   - We use Yaw, pitch and roll as input to the network. 
   
4. Pupil Location
   -  Location: X,Y
   - it helps to  approximate the gaze direction
   
- Total Number of Features: 10 

- Eye Branch
1. Left Eye Image:
   - Image dim: 36x60x3
   - Extracted by fitting a bounding around facial key points extracted in face detection step

  
#### CNN features (Not Using)
- Script to extract CNN features from video frames using a pre-trained VGG-19 model.
- Code: DGAZE/Codes/Extract_Features_Codes/CNN_face_features/extract_features.sh


## Training the model:
- Check the training data saved in pkl files using the scipt: DGAZE/Codes/Training_Model/Check_training_data.ipynb
- Train the model by running: python main.py