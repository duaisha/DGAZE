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


        
def get_align_videos(path, driver, sample_no, driver_view_out, road_view_out, drop):
    '''
    func to obtain road view by matching frames from driver view, 
    road view and projected road view to get the same number of frames in each.
    steps:
        -- get number of frames in driver view, projected road view and actual road view
        -- Drop extra frames using (drop = larger_frame_count/diff) to get equal number of frames in each video
    '''
    driver_view = os.path.join(path, 'dataset_download', driver, 'driver_view', 'sample' +\
                                                                       str(sample_no) +'.avi')            
    road_view = os.path.join(path,  'dataset_download', 'road_view', 'trip' + str(sample_no)\
                                                                     + '_out_hist.avi')
              
    if sample_no < 10:
        gt_point = os.path.join(path, 'dataset_download', 'road_view', 'trip' + str(sample_no) + '_out.npy')
        gt_point = np.load(gt_point)
    else:
        gt_point = os.path.join(path, 'dataset_download', 'road_view', 'trip' + str(sample_no) + '_out.txt')
        gt_point = pd.read_csv(gt_point, header = None).values   
                         
    cap_driver_view = cv2.VideoCapture(driver_view) # read driver view
    nframes_driver_view = get_frame_count(driver_view) # frame count of driver view video
    
    cap_road_view = cv2.VideoCapture(road_view) # read road view
    nframes_road_view = get_frame_count(road_view) # frame count of road view video
   
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    driver_view_out = os.path.join(path, 'dataset', driver, 'driver_view', 'sample' + str(sample_no) +'.avi')
    driver_view_out = cv2.VideoWriter(driver_view_out, fourcc, 25, (1440,1080)) 
    
    road_view_out = os.path.join(path,  'dataset', driver, 'road_view', 'sample' + str(sample_no) +'.avi')
    road_view_out = cv2.VideoWriter(road_view_out, fourcc, 25, (1920, 1080)) 
    
    road_gaze_point_out = os.path.join(path,  'dataset', driver, 'road_view', 'sample' + str(sample_no) +'.npy')
    
    i = 0; count = 0; k = 0; gaze_point_out = []

    nframes = nframes_driver_view
    
    while(i < nframes):
        i += 1     
        if(drop != -1 and i%drop == 0):
            if(nframes_driver_view > nframes_road_view):
                ret1, frame1 = cap_driver_view.read()
            else:
                ret3, frame3 = cap_road_view.read()
                k +=1
        else:
            ret1, frame1 = cap_driver_view.read()
            ret3, frame3 = cap_road_view.read()
            if(ret1 == False or ret3 == False):
                break
            gaze_point_out.append(gt_point[k])
            k +=1
            count += 1
            driver_view_out.write(frame1)
            road_view_out.write(frame3)
            
              
    gaze_point_out = np.array(gaze_point_out)
    np.save(road_gaze_point_out, gaze_point_out)
        
    driver_view_out.release(); road_view_out.release();
    cap_driver_view.release(); cap_road_view.release()
             

        
def dataset(path):
    
    assert os.path.exists(path), "path does not exists!"
        
    drivers = os.listdir(path + 'dataset_download/')
    
    for driver in drivers:
        
        if driver == 'road_view':
            continue;
        
        print(f'--> Processing for {driver}')
        
        nsamples = len(os.listdir(os.path.join(path, 'dataset_download', driver, 'driver_view')))
        
        driver_view_out = os.path.join(path, 'dataset', driver, 'driver_view')
        road_view_out = os.path.join(path, 'dataset', driver, 'road_view')
        
        if os.path.exists(driver_view_out) and os.path.exists(road_view_out):
            continue
        
        if not os.path.exists(driver_view_out):
            os.makedirs(driver_view_out, exist_ok=True)
        
        if not os.path.exists(road_view_out):
            os.makedirs(road_view_out, exist_ok=True)

        drop_data = pd.read_csv(os.path.join(path, 'dataset_download', driver, 'drop_rate_per_video.csv'))
        samples = drop_data['sample_no']
        drop_rate = drop_data['drop_rate'].astype(int)

        for sample_no in tqdm(range(1, nsamples+1)):
            drop = drop_rate[np.where(samples == 'sample'+str(sample_no)+'.avi')[0][0]]
            get_align_videos(path, driver, sample_no, driver_view_out, road_view_out, drop)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Creates a folder name dataset in DGAZE repo, dataset contains driver video, road video and gaze point on road video!")

    parser.add_argument('-path', '--path', type=str, required=True,
                        help="path to DGAZE repo containing dataset_download")

    args = parser.parse_args()

    dataset(args.path)