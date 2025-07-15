import sys
import os
import glob
import numpy as np
from sklearn.cluster import KMeans

from natsort import natsorted
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

sub, run = str(sys.argv[1]), str(sys.argv[2])

def flatten_img(file_path):
    """
    Read plotted gaze path and convert it to a flattend
    numpy array.
    """
    f_img = Image.open(file_path).convert("L")
    f_arr = np.asarray(f_img)
    return f_arr.flatten()

def create_dirs(sub, metric):
    """
    Create output dirs for further analysis-
    
    Return:
        string of output path
    """
    output_dir= os.path.join("/home", "data", "study_gaze_tracks", "derivatives", f"{metric}-distance-models", f"attention-mode", f"sub-{sub}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

hm_list = natsorted(glob.glob(os.path.join("/home", "data", "study_gaze_tracks", "scratch", f"spatial-attention_heatmaps", 
                                f"sub-{sub}_output_fixation-density-maps", f"sub-{sub}_run-{run}_*.png")))

vec_container = list(map(flatten_img, hm_list))

# initialise cluster alg.
kmeans = KMeans(n_clusters=2)
kmeans.fit(vec_container)