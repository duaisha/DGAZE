import os, sys
import numpy as np
import pandas as pd
import cv2 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def get_gaze_point(driver, seq, object_annot):
    if seq < 9:
        temp = np.ones((object_annot.shape[0],4))*-1
        return np.hstack((object_annot,temp))
    else:
        bbox = object_annot
        pt1 = bbox[:,1] + (bbox[:,3] - bbox[:,1])/2
        pt2 = bbox[:,2] + (bbox[:,4] - bbox[:,2])/2
        pt1 = np.expand_dims(pt1, axis =1)
        pt2 = np.expand_dims(pt2, axis =1)
        return np.hstack((pt1.astype(int), pt2.astype(int), object_annot[:,1:]))
    
def get_driver_features(data_path, driver, sequences, frames_per_seq):
    """
    Load driver features for DGAZE:
    Input: Images
    Output: left_eye features of dim = (nframes x 36 x 60 x 3)
            right_eye features of dim = (nframes x 36 x 60 x 3)
            face location features of dim = (nframes x 4)
            # face location means driver face location inside cars
            headpose_pupil is of dim = (nframes x 11)
            # (nframes, roll, pitch, yaw, lpupil(x,y), rpupil(x,y), face_area, nose(x,y))
            # x,y left eye pupil location
            gaze_point is ground truth of dim  = (nframesx6)
            # First two values are x,y for point annotation(center of object)
            # Next four are x,y corresponding to top left point of the object and bottom right of the object respectively.
    """

    driver_features = {}
    total_frames = 0
    
    driver_features_path = data_path + '/' + driver
    eye_features_path = driver_features_path + "/explicit_face_features_game/"
    face_features_path = driver_features_path + "/explicit_face_points/"
    annot_path = driver_features_path +"/original_road_view/" 
    

    if os.path.exists(eye_features_path) and os.path.exists(annot_path):
        for seq in tqdm(range(0, sequences)):
            features = {}

            eye_features = eye_features_path + "sample" + str(seq+1)
            face_location_features = face_features_path + "sample_" + str(seq+1)
            annot = annot_path + "sample_" + str(seq+1) + ".npy"

            left_eye_features = eye_features + "_left_eye_data.npy"
            right_eye_features = eye_features + "_right_eye_data.npy"
            headpose_pupil_features = eye_features + "_headpose_pupil.npy"
            face_location_features = face_location_features +"_face_points.npy"
            
            try:
                if os.path.exists(left_eye_features) and os.path.exists(right_eye_features) and \
                os.path.exists(headpose_pupil_features) and os.path.exists(face_location_features):
                    if np.load(left_eye_features, allow_pickle=True).shape[0] == get_gaze_point(driver, seq, np.load(annot, allow_pickle=True)).shape[0]:
                        features['left_eye']  = np.load(left_eye_features, allow_pickle=True)
                        features['right_eye'] = np.load(right_eye_features, allow_pickle=True)
                        features['headpose_pupil']  = np.load(headpose_pupil_features, allow_pickle=True)
                        features['face_location'] = np.load(face_location_features, allow_pickle=True)
                        features['gaze_point'] = get_gaze_point(driver, seq, np.load(annot, allow_pickle=True))
                        total_frames += features['gaze_point'].shape[0]

                        driver_features["".join(['seq',str(seq+1)])] = features
            except:
                print(f"Error: Cannot load for driver {driver} sequence {seq}")
       
        driver_features['frames_count'] = total_frames
        
    return driver_features

def get_data(data_path, drivers, sequences, frames_per_seq):
    data = {}
    for driver in drivers:
        data["driver"+driver.split('user')[-1]] = get_driver_features(data_path, driver, sequences, frames_per_seq)
    return data



