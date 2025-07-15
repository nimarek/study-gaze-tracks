import os
import sys
import glob
from itertools import product

import numpy as np
import pandas as pd

import cv2
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import seaborn as sns

run = sys.argv[1]

# chunk list for 4 sec (scene onset)
# chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

# chunk list for 4 sec (complete)
chunk_list = [0, 253, 252, 242, 275, 249, 243, 306, 178]

def create_dirs(run):    
    output_dir= os.path.join("/home", "data", "study_gaze_tracks", "derivatives", "spatial-attention_scene-complete_ssim", f"run-{run}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def struct_sim(run, chunk_list):
    """
    Read images according to event file
    """
    event_input = pd.read_csv(os.path.join("/home", "data", "study_gaze_tracks", "code", "reference_spatial-attention", f"complete_ses-movie_task-movie_run-{run}_events.tsv"), sep="\t", index_col=False)

    ssim_list, chunk_max = [], chunk_list[int(run)]
    
    for chunk_a, chunk_b in product(event_input["trial_type"], repeat=2):
        print(chunk_a, chunk_b)
        input_img_path_a = glob.glob(os.path.join("/home", "data", "study_gaze_tracks", "scratch", "ssim_movie-frames", f"raw_run-{run}", f"{chunk_a}", f"average-img_chunk-chunk-*.png"))
        input_img_path_b = glob.glob(os.path.join("/home", "data", "study_gaze_tracks", "scratch", "ssim_movie-frames", f"raw_run-{run}", f"{chunk_b}", f"average-img_chunk-chunk-*.png"))

        gray1 = cv2.cvtColor(cv2.imread(input_img_path_a[0]), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.imread(input_img_path_b[0]), cv2.COLOR_BGR2GRAY)  
        ssim_list.append(ssim(gray1, gray2))
    return np.reshape(ssim_list, (chunk_max, chunk_max))

def plot_heatmap(run, labels, df, output_heat_maps):
    # masked_tria = np.triu(df)
    hm_fig_dissim = sns.heatmap(df, xticklabels=labels, yticklabels=labels, 
                         cmap="RdBu_r") # , mask=masked_tria
    hm_fig_dissim.set_title(f"run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"ssim_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

print(f"starting run-{run}")
output_dir = create_dirs(run)
struct_sim_df = struct_sim(run, chunk_list)
plot_heatmap(run, list(range(chunk_list[int(run)])), struct_sim_df, output_dir)
pd.DataFrame(struct_sim_df).to_csv(output_dir + f"/ssim_run-{run}.tsv", header=False, index=False)    