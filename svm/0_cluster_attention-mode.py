import os
import glob
import numpy as np
import pandas as pd

from natsort import natsorted
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

sub_list = ["01", "02", "03", "04", "05", "06", "09", 
            "10", "14", "15", "16", "17", "18", "19", 
            "20"]
run_list = list(range(1, 9))

def load_data(sub, run, metric):
    rdm_list = natsorted(glob.glob(os.path.join("/home", "data", "study_gaze_tracks",
                                                "derivatives",
                                                f"{metric}-distance-models", 
                                                "spatial-attention_attention-mode", 
                                                f"sub-{sub}", 
                                                f"sub-{sub}_sub-*_run-{run}_*.tsv")))
    return rdm_list

def create_dirs(sub, metric):
    """
    create output dirs for further analysis.
    
    Return:
        string of output path
    """
    output_dir= os.path.join("/home", "data", "study_gaze_tracks",
                            "derivatives", 
                            f"{metric}-distance-models", 
                            f"attention-mode_cluster-analysis", 
                            f"sub-{sub}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def prepare_analysis(sub, run):
    """
    calculate average matrix from all 
    input matrices per subject.
    """
    df_container = []
    df_list = load_data(sub=sub, run=run, metric="euclidean")
    for df_path in df_list:
        df_container.append(np.loadtxt(df_path, delimiter=","))
    return np.array(df_container).mean(0)

def plot_heatmap_eucl(sub, run, df):
    
    output_dir = create_dirs(sub, metric="euclidean")
    hm_fig_dissim = sns.heatmap(df, cmap="RdBu_r")
    hm_fig_dissim.set_title(f"run-{run}", fontweight="bold")
    plt.savefig(output_dir + f"/sub-{sub}_run-{run}_combined-matrix.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return output_dir

def cluster_all(sub_list, run_list):
    """
    start cluster analysis. Combine labels
    for all runs within-sub for fmri svm.
    """
    for sub in sub_list:
        for run in run_list:
            df = prepare_analysis(sub=sub, run=run)
            median_distance = np.median(df)
            output_dir = plot_heatmap_eucl(sub=sub, run=run, df=df)
            # label_df = pd.DataFrame(hyp.cluster(df, n_clusters=None, cluster={"model" : "AgglomerativeClustering", "distance_threshold" : median_distance}), columns=["label"])
            
            cluster_df = AgglomerativeClustering(n_clusters=None, distance_threshold=median_distance).fit(df)
            label_df = pd.DataFrame(cluster_df.labels_, columns=["label"])
            # add run-column to df
            run_column = [f"run-{run}" for x in range(1, len(cluster_df.labels_)+1)]
            label_df["run"] = run_column

            trial_column = [x for x in range(1, len(cluster_df.labels_)+1)]
            label_df["trial"] = trial_column
            label_df.to_csv(output_dir + f"/sub-{sub}_run-{run}_labels.tsv", sep="\t", index=False)

            # hyp.plot(df, ".", hue=label_df, title=f"sub-{sub} run-{run}", save_path=output_dir + f"/sub-{sub}_run-{run}")
        
        # combine dfs
        combine_labels(sub=sub, out_dir=output_dir)
    return None

def combine_labels(sub, out_dir, metric="euclidean"):
    df_list = []
    data_list = natsorted(glob.glob(os.path.join("/home", "data", "study_gaze_tracks",
                                                "derivatives", 
                                                f"{metric}-distance-models", 
                                                f"attention-mode_cluster-analysis", 
                                                f"sub-{sub}",
                                                f"sub-{sub}_run-*_labels.tsv")))
    for df in data_list:
        df_tmp = pd.read_csv(df, sep="\t")
        df_list.append(df_tmp)
        
    comb_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"sub-{sub} label-0:\t", np.count_nonzero(comb_df["label"] == 0))
    print(f"sub-{sub} label-1:\t", np.count_nonzero(comb_df["label"] == 1))
    comb_df.to_csv(out_dir + f"/sub-{sub}_combined-labels.tsv", sep="\t")

# run actual analysis
cluster_all(sub_list=sub_list, run_list=run_list)