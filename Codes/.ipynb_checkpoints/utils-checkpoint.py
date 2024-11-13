## Utility Functions
import matplotlib.patches as patches

def get_metadata(data_dict):
    """
    Retrieve metadata from a dictionary.

    Parameters:
        data_dict (dict): Dictionary to extract metadata from.

    Returns:
        tuple: Original dictionary and a list of its keys.
    """
    return data_dict, list(data_dict.keys())

def print_metadata(driver_data, metakeys):
    """
    Print metadata information for drivers, sequences, and features based on specified keys.

    Parameters:
        driver_data (dict): Dictionary containing driver data.
        metakeys (list): List of metadata types to print (e.g., 'drivers', 'sequences', 'features').
    """
    drivers_dict, drivers = get_metadata(driver_data)
    seq_dict, sequences = get_metadata(drivers_dict[drivers[0]])
    features_dict, features = get_metadata(seq_dict[sequences[0]])

    if 'drivers' in metakeys:
        print("List of Drivers: \n{}\n".format(drivers))
    if 'sequences' in metakeys:
        print("List of Sequences: \n{}\n".format(sequences))
    if 'features' in metakeys:
        print("List of Features: \n{}\n".format(features))

def get_dgaze_frames_count(driver_data, drivers):
    """
    Calculate and print the total frames count in the DGAZE dataset.

    Parameters:
        driver_data (dict): Dictionary containing driver data.
        drivers (list): List of driver identifiers.
    """
    total_frames = 0
    for driver in drivers:
        frames_count = driver_data[driver].get('frames_count', 0)
        total_frames += frames_count
        print("Frames count for driver {}: {}".format(driver, frames_count))

    print("\nTotal frames in DGAZE dataset: {}".format(total_frames))
    
    

# def plot_gaze_points(data_path , gaze_point):
#     cap = cv2.VideoCapture(data_path + 'user12/original_road_view/sample_56.avi')
#     ret, frame = cap.read()
    
#     y = np.where(gaze_point[:,0]>=1920)
#     gaze_point[y[0], 0]=1919
#     y = np.where(gaze_point[:,0]<0)
#     gaze_point[y[0],0]=0
#     y = np.where(gaze_point[:,1]>=1080)
#     gaze_point[y[0], 1]=1080
#     y = np.where(gaze_point[:,1]<0)
#     gaze_point[y[0], 1]=0

#     plt.figure()
#     plt.imshow(frame)
#     plt.scatter(gaze_point[:,0], gaze_point[:,1], c='r')

# plot_gaze_points(data_path, train['gaze_point'])