# DGAZE: Driver Gaze Mapping on Road

DGAZE is a new dataset for mapping the driver's gaze onto the road!

## Important Links
- **[Link to Paper](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/DGAZE/paper.pdf)**  
   This paper explains the research and methodology behind DGAZE.
  
- **[Project Webpage](https://cvit.iiit.ac.in/research/projects/cvit-projects/dgaze)**  
   Visit the official project webpage for more details about DGAZE and its applications.

- **[Extracted Features (Google Drive)](https://drive.google.com/drive/folders/12L2ctyKjOcDl8-oZ8jgoDihnFVBjIbMj?usp=sharing)**  
   You can download the extracted features used in the DGAZE research from the link above.

![image](https://github.com/user-attachments/assets/3ef3cd0a-22b4-41c6-8381-2b1efc73e1dc)
![image](https://github.com/user-attachments/assets/edd02f59-b2ac-4418-a31c-834a02c8ef59)


## Clone DGAZE Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/duaisha/DGAZE.git
   ```
2. Navigate to the DGAZE directory:
   ```bash
   cd DGAZE
   ```

Note that Step-1 and Step-2 is to use raw dataset.In order to use extracted features, switch directly to step-3

## Step-1: Prepare Dataset

### 1. Download DGAZE Dataset
- Download the dataset from the provided link in the DGAZE repository. It will download a folder named **"dataset_download"** (approximately 20GB).
- Place this folder in the DGAZE repository directory.

### 2. Code for Drop Rate Calculation (Reference Code)
- The drop rate for each sample corresponding to each driver is already provided, so this step can be skipped.
- If you want to calculate it yourself, use the following command:
   ```bash
   python Codes/Dataset_codes/drop_rate.py --path "<Path_to_DGAZE_folder>"
   ```

### 3. Get Dataset Folder
- Run the following command to retrieve the dataset folder:
   ```bash
   python Codes/Dataset_codes/dataset.py --path "<Path_to_DGAZE_folder>"
   ```

## Visualize the DGAZE Dataset
- Use the `visualize_dataset.ipynb` file located in **Codes/Dataset_codes/** to visualize the driver view and road view with gaze point variation across the sample video for each driver.

## Step 2: State-of-the-Art (SOTA) Feature Extraction Methods Used:

### Feature Branch
1. **Face Detection**:
   - Use the **DLIB** library to extract facial landmarks and the face bounding box.
   - This helps in determining the location of the face in the scene using the bounding box and nose.

2. **Area of Face Bounding Box**:
   - Calculate the area of the face bounding box. This is important for estimating the distance of the head from the screen, assuming that facial area remains consistent across drivers.

3. **Head Pose**:
   - Use **yaw**, **pitch**, and **roll** angles as input features for the network.

4. **Pupil Location**:
   - Use the X and Y coordinates to approximate the gaze direction.

   **Total Number of Features**: 10

### Eye Branch
1. **Left Eye Image**:
   - Image dimensions: **36x60x3**
   - Extracted by fitting a bounding box around facial key points detected in the face detection step.

### CNN Features (Not Used)
- A script is available to extract CNN features from video frames using a pre-trained VGG-19 model.
   ```bash
   DGAZE/Codes/Extract_Features_Codes/CNN_face_features/extract_features.sh
   ```

## Step-3: Extracted Features
- You can either download the pre-extracted features used in the research or generate your own following the steps in the next section.

### 1. Download Extracted Features
- Download the extracted features from Google Drive:
   - **DGAZE_extracted_data.pkl**
   - **DGAZE_data_split.pkl**

- Place these downloaded `.pkl` files in the **DGAZE/DGAZE_extracted_data** directory.

### 2. Contents of 'DGAZE_extracted_data.pkl':
   - **Left Eye Image**: Shape (nframes x 36 x 60 x 3)
   - **Right Eye Image**: Shape (nframes x 36 x 60 x 3)
   - **Face Location / Bounding Box**: Shape (nframes x 4)
   - **Head Pose & Pupil Data**: Shape (nframes x 11)
     - Includes: (nframes, roll, pitch, yaw, lpupil(x,y), rpupil(x,y), face_area, nose(x,y))
   - **Gaze Point**: Shape (nframes x 6)
     - First two values: center of the object bounding box (x, y)
     - Next four values: coordinates of the top-left and bottom-right corners of the object



## Step-4: Training the Model
1. Check the training data saved in `.pkl` files using the following script:
   ```bash
   DGAZE/Codes/Training_Model/Check_training_data.ipynb
   ```
2. Train the model by running:
   ```bash
   python main.py
   ```

