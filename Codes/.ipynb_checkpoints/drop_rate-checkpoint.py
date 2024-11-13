import os
import cv2
import argparse

import pandas as pd
import numpy as np
import dlib
from tqdm import tqdm 


def get_frame_count(input_video):
    '''
    func to count number of frames in input video
    ''' 
    count = 0
    vid = cv2.VideoCapture(input_video)
    while(True):
        ret, frame = vid.read()
        if(ret == False):
            break
        count += 1
    return count



def get_drop_rate(path, driver, sample_no):
    '''
    func to obtain road view by matching frames from driver view, 
    road view and projected road view to get the same number of frames in each.
    steps:
        -- get number of frames in driver view, projected road view and actual road view
        -- Drop extra frames using (drop = larger_frame_count/diff) to get equal number of frames in each video
    '''
    driver_view = os.path.join(path, 'dataset_download', driver, 'driver_view', 'sample' + str(sample_no) +'.avi')            
    projected_road_view = os.path.join(path, 'extras', driver, 'projected_road_view', 'sample' + str(sample_no)+'.avi')  
    road_view = os.path.join(path,  'dataset_download', 'road_view', 'trip' + str(sample_no) + '_out_hist.avi')
              
    if sample_no < 10:
        gt_point = os.path.join(path, 'dataset_download', 'road_view', 'trip' + str(sample_no) + '_out.npy')
    else:
        gt_point = os.path.join(path, 'dataset_download', 'road_view', 'trip' + str(sample_no) + '_out.txt')
      
                                     
    cap_driver_view = cv2.VideoCapture(driver_view) # read driver view
    nframes_driver_view = get_frame_count(driver_view) # frame count of driver view video
    
    cap_projected_road_view = cv2.VideoCapture(projected_road_view) # read projected_road view
    nframes_projected_road_view = get_frame_count(projected_road_view) # frame count of projected_road view video
    
    cap_road_view = cv2.VideoCapture(road_view) # read road view
    nframes_road_view = get_frame_count(road_view) # frame count of road view video

    flag =0; i = 0; count = 0; k = 0; gaze_point_out = []

    if(nframes_projected_road_view == nframes_road_view):
        flag = 1
        nframes = nframes_projected_road_view
        first = nframes_projected_road_view
        second = nframes_road_view

    elif(nframes_road_view < nframes_projected_road_view):
        nframes = nframes_projected_road_view 
        first = nframes_projected_road_view 
        second = nframes_road_view 

    elif(nframes_road_view > nframes_projected_road_view):
        nframes = nframes_road_view
        first = nframes_road_view
        second = nframes_projected_road_view

    diff = first - second
    if(diff != 0):
        drop = np.round(first/diff)
    else:
        drop = -1
    
    return drop
      
        
def drop_rate_per_sample(path):
    
    assert os.path.exists(path), "path does not exists!"
        
    drivers = os.listdir(path + 'dataset_download/')
    
    for driver in drivers:
        if driver == 'road_view':
            continue
        
        print(f'--> Processing for {driver}')
        
        nsamples = len(os.listdir(os.path.join(path, 'dataset_download', driver, 'driver_view')))
        
        if not os.path.exists(os.path.join(path, 'dataset_download', driver, 'drop_rate_per_video.csv')):
            drop_rate_file = open(os.path.join(path, 'dataset_download', driver, 'drop_rate_per_video.csv'),"w")
            drop_rate_file.write('sample_no' + ',' + 'drop_rate' +"\n")

            for sample_no in tqdm(range(1, nsamples+1)):
                drop = get_drop_rate(path, driver, sample_no)
                drop_rate_file.write('sample'+str(sample_no)+'.avi' + ',' + str(drop)+"\n")

            drop_rate_file.close()

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Since driver video and actual road video are collected at different fps,\
                we compute drop rate for each sample corresponding to all drivers")

    parser.add_argument('-path', '--path', type=str, required=True,
                        help="path to DGAZE repo containing dataset_download")

    args = parser.parse_args()

    drop_rate_per_sample(args.path)