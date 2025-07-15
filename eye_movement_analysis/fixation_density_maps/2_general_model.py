import os
import glob

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

sigma = "8"
in_distance = "sqeuclidean"
root_dir = f"/home/data/study_gaze_tracks/derivatives/{in_distance}-distance-models"

def plot_heatmap_eucl(run, df, output_heat_maps):
    # masked_tria = np.triu(df)
    hm_fig_dissim = sns.heatmap(df, # vmin=0, vmax=2,
                         cmap="RdBu_r") # , mask=masked_tria
    hm_fig_dissim.set_title(f"run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"/general-model_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

for run in range(1, 9):
    print("starting with run:\t", run)
    input_data_list = []

    for model in glob.glob(root_dir + f"/spatial-attention_sigma-{sigma}/*/*run-{run}_matrix.tsv"):
        print(f"working with data:\t {model}")
        model_rdm = np.loadtxt(model, delimiter=",")
        input_data_list.append(model_rdm)

        # average over all participants per run
        general_model = np.array(input_data_list).mean(0)
        general_model = np.median(input_data_list, axis=0)

        # save general model per run
        output_dir = root_dir + f"/spatial-attention_sigma-{sigma}/general_model"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    plot_heatmap_eucl(run, general_model, output_dir)
    pd.DataFrame(general_model).to_csv(output_dir + f"/run-{run}_general-model-matrix.tsv", header=False, index=False)