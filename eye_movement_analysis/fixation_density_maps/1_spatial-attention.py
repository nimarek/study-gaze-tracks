import sys
import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns

soi = str(sys.argv[1])
run = str(sys.argv[2])

sigma, metric = "8", "sqeuclidean"

# chunk list for 4 sec (scene onset)
# chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

# chunk list for 4 sec (complete)
chunk_list = [0, 253, 252, 242, 275, 249, 243, 306, 178]

def flatten_img(file_path):
    """
    Read plotted gaze path and convert it to a flattend
    numpy array.
    """
    f_img = Image.open(file_path).convert("L") # "L" for greyscale
    f_arr = np.asarray(f_img)
    return f_arr.flatten()

def create_dirs(sub, sigma=None, metric="sqeuclidean"):
    """
    Create output dirs for further analysis-
    
    Return:
        string of output path
    """
    output_dir= os.path.join("/home", "data", "study_gaze_tracks", "derivatives", f"{metric}-distance-models", f"spatial-attention_sigma-{sigma}", f"sub-{sub}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_heatmap(sub, run, df, output_heat_maps):
    hm_fig_dissim = sns.heatmap(df, cmap="RdBu_r", yticklabels=False, xticklabels=False)
    
    plt.savefig(output_heat_maps + f"/sub-{sub}_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

def start_analysis(soi, run, chunk_max, sigma=None, metric="eucldiean"):
    """
    (1) Calculate average vector per chunk for all 
    subjects except the subject of interest (SOI).
    (2) Load heatmap for SOI and call flatten_img.
    """
    output_dir = create_dirs(soi, sigma=sigma, metric=metric)    
    correlation_container = []

    outf_path = output_dir + f"/sub-{soi}_run-{run}_matrix.tsv"
    if os.path.isfile(outf_path):
        raise ValueError("file already exists, abort mission ...")

    for chunk_a in range(1, chunk_max+1):
        chunk_a_heatmap = os.path.join("/home", "data", "study_gaze_tracks", "scratch", 
                            f"spatial-attention_scene-complete_heatmaps", 
                            f"sub-{soi}_output_fixation-density-maps", 
                            f"sub-{soi}_run-{run}_chunk-{chunk_a}_sigma-{sigma}.png")
        
        chunk_a_arr = flatten_img(chunk_a_heatmap)
        
        for chunk_b in range(1, chunk_max+1):            
            chunk_b_heatmap = os.path.join("/home", "data", "study_gaze_tracks", "scratch", 
                                        f"spatial-attention_scene-complete_heatmaps", 
                                        f"sub-{soi}_output_fixation-density-maps", 
                                        f"sub-{soi}_run-{run}_chunk-{chunk_b}_sigma-{sigma}.png")
            
            # print(f"chunk-A: \t {chunk_a_heatmap}")
            # print(f"chunk-B: \t {chunk_b_heatmap}")

            chunk_b_arr = flatten_img(chunk_b_heatmap)
            print(np.around(distance.cdist([chunk_a_arr], [chunk_b_arr], metric)[0][0], decimals=3)) # / 33
            correlation_container.append(np.around(distance.cdist([chunk_a_arr], [chunk_b_arr], metric)[0][0], decimals=3)) # / 33
            
    spatial_model = np.reshape(correlation_container, (chunk_max, chunk_max))
    plot_heatmap(sub=soi, run=run, df=spatial_model, output_heat_maps=output_dir)
    pd.DataFrame(spatial_model).to_csv(outf_path, header=False, index=False)
    return None


print(f"starting soi-{soi} run-{run} ...")
chunk_max = chunk_list[int(run)]
start_analysis(soi=soi, run=run, chunk_max=chunk_max, sigma=sigma, metric=metric)