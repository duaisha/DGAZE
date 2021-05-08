#!/bin/bash
clear

input_dir="/ssd_scratch/cvit/isha2/DGM_final2/dataset_samples_callibrated/user14/driver_view_cropped/"
output_dir="/ssd_scratch/cvit/isha2/DGM_final2/dataset_samples_callibrated/user14/driver_features/"

python3 extract_features.py -i $input_dir -o $output_dir -m "vggface"
