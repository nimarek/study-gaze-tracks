import os
import glob
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.spatial import distance

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

# chunk list for 4 sec
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]
metric_analysis = "euclidean"

def create_dirs(run):    
    output_dir= os.path.join("/home", "data", "study_gaze_tracks", "derivatives", "spatial-attention-duration-4_ssim", f"run-{run}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def flatten_img(file_path, bins):
    """
    Read plotted gaze path and convert it to a flattend
    numpy array. Zscore and order the array.
    
    Return:
        Digitized / z-scored numpy array.
    """
    file = Image.open(file_path).convert("L")
    file = np.stack((file,) * 3, axis=-1)
    file = np.array(file) / 255.0
    return np.digitize(zscore(file.flatten()), np.arange(bins))

def mean_vector(file_path_list, bins):
    """
    Calculate mean vector from given list of movie frames.
    
    Return:
        numpy array containing average vector.
    """
    heatmap_container = []
    for path in file_path_list:
        heatmap_container.append(flatten_img(path, bins))
    return np.mean(np.array(heatmap_container), axis=0)

def img_sim(run, chunk_list, metric):
    """
    Read images according to event file
    """
    event_input = pd.read_csv(os.path.join("/home", "data", "study_gaze_tracks", "code", "reference_spatial-attention", f"ses-movie_task-movie_run-{run}_events.tsv"), sep="\t", index_col=False)
    img_sim_list, chunk_max = [], chunk_list[int(run)]
    
    for onset_a, onset_b in product(event_input["onset"], repeat=2):
        input_img_path_a = glob.glob(os.path.join("/home", "data", "study_gaze_tracks", "scratch", "spatial-attention_movie-frames", f"raw_run-{run}", f"*run-{run}*onset-{int(onset_a)}.0.jpg"))
        input_img_path_b = glob.glob(os.path.join("/home", "data", "study_gaze_tracks", "scratch", "spatial-attention_movie-frames", f"raw_run-{run}", f"*run-{run}*onset-{int(onset_b)}.0.jpg"))
        
        img_a = mean_vector(input_img_path_a, bins=40)
        img_b = mean_vector(input_img_path_b, bins=40)

        img_sim_list.append(np.around(distance.cdist([img_a], [img_b], metric)[0][0], decimals=3))
    return np.reshape(img_sim_list, (chunk_max, chunk_max))

def plot_heatmap(run, labels, df, output_heat_maps):
    # masked_tria = np.triu(df)
    hm_fig_dissim = sns.heatmap(df, xticklabels=labels, yticklabels=labels, 
                         cmap="RdBu_r") # , mask=masked_tria
    hm_fig_dissim.set_title(f"run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"ssim_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

for run in range(1, 9):
    print(f"starting run-{run}")
    output_dir = create_dirs(run)
    img_sim_df = img_sim(run, chunk_list, metric=metric_analysis)
    plot_heatmap(run, list(range(chunk_list[int(run)])), img_sim_df, output_dir)
    pd.DataFrame(img_sim_df).to_csv(output_dir + f"/ssim_run-{run}.tsv", header=False, index=False)    