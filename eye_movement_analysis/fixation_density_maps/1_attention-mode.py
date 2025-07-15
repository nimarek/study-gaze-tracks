import sys
import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from scipy.stats import zscore
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns

soi, sub_comp = str(sys.argv[1]), str(sys.argv[2])
run = str(sys.argv[3])

metric_list = ["euclidean", "correlation"]

# chunk list for 4 sec
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

# def flatten_img(file_path, bins):
#     """
#     Read plotted gaze path and convert it to a flattend
#     numpy array.
    
#     Return:
#         Digitized / z-scored numpy array.
#     """
#     file = Image.open(file_path).convert("L")
#     file = np.stack((file,) * 3, axis=-1)
#     file = np.array(file) / 255.0
#     return np.digitize(file.flatten(), np.arange(bins))

def flatten_img(file_path):
    """
    Read plotted gaze path and convert it to a flattend
    numpy array.
    """
    f_img = Image.open(file_path).convert("L") # "L" for greyscale
    f_arr = np.asarray(f_img)
    return f_arr.flatten()

def create_dirs(sub, metric):
    """
    Create output dirs for further analysis-
    
    Return:
        string of output path
    """
    output_dir= os.path.join("/home", "data", "study_gaze_tracks", "derivatives", f"{metric}-distance-models", f"spatial-attention_attention-mode", f"sub-{sub}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_heatmap(sub, sub_comp, run, df, output_heat_maps):

    hm_fig_dissim = sns.heatmap(df, cmap="RdBu_r")
    
    hm_fig_dissim.set_title(f"sub-{sub} run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"/sub-{sub}_sub-{sub_comp}_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

def start_analysis(soi, sub_comp, run, chunk_max, metric="eucldiean"):
    """
    (1) Calculate average vector per chunk for all 
    subjects except the subject of interest (SOI).
    (2) Load heatmap for SOI and call flatten_img.
    """
    output_dir = create_dirs(soi, metric=metric)    
    comp_container, correlation_container = [], []
    # soi_ident = f"sub-{soi}"

    outf_path = output_dir + f"/sub-{soi}_sub-{sub_comp}_run-{run}_matrix.tsv"
    if os.path.isfile(outf_path):
        raise ValueError("file already exists, abort mission ...")

    for chunk_target in range(1, chunk_max+1):
        soi_heatmap = os.path.join("/home", "data", "study_gaze_tracks", "scratch", 
                                   f"spatial-attention_heatmaps", 
                                   f"sub-{soi}_output_fixation-density-maps", 
                                   f"sub-{soi}_run-{run}_chunk-{chunk_target}.png")

        soi_arr = flatten_img(soi_heatmap)

        for chunk_compare in range(1, chunk_max+1):
            comp_heatmap = os.path.join("/home", "data", "study_gaze_tracks", "scratch", 
                                        f"spatial-attention_heatmaps", 
                                        f"sub-{sub_comp}_output_fixation-density-maps", 
                                        f"sub-{sub_comp}_run-{run}_chunk-{chunk_compare}.png")


            sub_comp_arr = flatten_img(comp_heatmap)
            correlation_container.append(np.around(distance.cdist([soi_arr], [sub_comp_arr], metric)[0][0], decimals=3))
    
    endo_exo_model = np.reshape(correlation_container, (chunk_max, chunk_max))
    plot_heatmap(sub=soi, sub_comp=sub_comp, run=run, df=endo_exo_model, output_heat_maps=output_dir)
    pd.DataFrame(endo_exo_model).to_csv(outf_path, header=False, index=False)
    return None

for metric in metric_list:
    print(f"starting soi-{soi} and sub-comp-{sub_comp}, run-{run} with metric-{metric}...")
    chunk_max = chunk_list[int(run)]
    start_analysis(soi=soi, sub_comp=sub_comp, run=run, chunk_max=chunk_max, metric=metric)