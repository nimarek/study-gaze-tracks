import sys
import os
import glob

import numpy as np
import pandas as pd

from PIL import Image
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import seaborn as sns

# I/O
sub, run = str(sys.argv[1]), str(sys.argv[2])
root_dir = "/home/data/study_gaze_tracks/derivatives/fix_maps/"
sigma = 0

# chunk list for 4 sec (scene onset)
# chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

# chunk list for 4 sec (complete)
chunk_list = [0, 253, 252, 242, 275, 249, 243, 306, 178]

print(f"number of scenes per run:\t {chunk_list[int(run)]}")

def load_split(sub, root_dir):
    """
    Function to load preprocessed eye-tracking data provided
    by Hanke et al. (https://www.studyforrest.org/)
    """
    file_path = root_dir + f"sub-{sub}_ses-movie_task-movie.npz"    
    with np.load(file_path, allow_pickle=True) as data:
        df_movie_frame = data["sa.movie_frame"]
        df_names = data["fa.name"].astype("str")
        df_samples = data["samples"]
        
    # create dataframe from arrays
    df_all = pd.DataFrame(data=df_samples, columns=df_names)
    df_all["frame"] = df_movie_frame
    return df_all

def flatten_img(file_path, bins=40):
    """
    Read plotted gaze path and convert it to a flattend
    numpy array. Zscore and order the array.
    """
    file = Image.open(file_path).convert("L")
    file = np.stack((file,) * 3, axis=-1)
    file = np.array(file) / 255.0
    return np.digitize(file.flatten(), np.arange(bins)) # 10

def chunk_data(df_all, b_frame, e_frame):
    chunked_df = df_all.loc[(df_all["frame"] >= b_frame) & (df_all["frame"] <= e_frame)]
    return chunked_df

def plot_norm_data(sub, run, chunk, chunk_data, sigma=None, output_path=os.getcwd()):
    """
    Note: This function purposefully overrwites existing 
    plots to save storage space.
    
    Return: Path to fixation density image.
    """
    width, height = 1280, 546
    
    extent = [0, width, height, 0]  # origin is the top left of the screen
    canvas = np.vstack((chunk_data["x"].to_numpy(), chunk_data["y"].to_numpy()))  # shape (2, n_samples)

    # bin into image-like format
    hist, _, _ = np.histogram2d(
        canvas[1, :],
        canvas[0, :],
        bins=(height, width),
        range=[[0, height], [0, width]]
    )

    # smooth the histogram
    hist = gaussian_filter(hist, sigma=sigma)

    # plot heatmap
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(
        hist,
        aspect="equal",
        cmap="Blues",
        origin="upper",
        alpha=1,
        extent=extent)
        
    save_path = output_path + f"/sub-{sub}_run-{run}_chunk-{chunk}_sigma-{sigma}.png"
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return save_path

def combine_data(chunks_run, data_list):
    """
    Convert numpy arrays back to pandas dataframe to
    leverage pandas nan friendly correlation function.
    """
    return pd.DataFrame(dict(zip(list(range(chunks_run)), data_list)))

def load_events(run, add_sec):
    """
    Load precalculated event files and use them as 
    onsets and duration to slice the eye-movement 
    data. And convert from seconds back to frames.
    
    Return: list of tuples with frame onsets and durations
    """
    event_path = f"/home/data/study_gaze_tracks/code/reference_spatial-attention/complete_ses-movie_task-movie_run-{run}_events.tsv"

    tmp_df = pd.read_csv(event_path, index_col=None, delimiter="\t")
    tmp_df["onset"] = tmp_df["onset"] + add_sec
    tmp_df["onset"] = tmp_df["onset"].apply(lambda x: x * 25) 
    tmp_df["duration"] = tmp_df["duration"].apply(lambda x: x * 25) # * 4
    tmp_df["offset"] = tmp_df["onset"] + tmp_df["duration"]
    return list(zip(tmp_df["onset"], tmp_df["offset"]))

def create_dirs(sub):
    output_gaze_maps = f"/home/data/study_gaze_tracks/scratch/spatial-attention_scene-complete_heatmaps/sub-{sub}_output_fixation-density-maps"
    if not os.path.exists(output_gaze_maps):
        os.makedirs(output_gaze_maps)
    return 0, output_gaze_maps

# load one example split as reference
exa_df = load_split(sub, root_dir)
start_frame, max_frame = np.min(exa_df["frame"]), np.max(exa_df["frame"])
print(f"starting frame: {start_frame}\t ending frame: {max_frame}")

# subtract 1, to account for fMRIPrep slice-timing
add_sec = -1
corr_container = []

# start plotting
output_dir, output_gaze_maps = create_dirs(sub)
steps_list = load_events(run, add_sec=add_sec)

chunk_num = 1
for start_time, end_time in steps_list:
    print(f"starting with chunk-{chunk_num}; onset frame: {start_time} offset frame: {end_time}")
    raw_input_df = load_split(sub=sub, root_dir=root_dir)
    chunked_df = chunk_data(df_all=raw_input_df, b_frame=start_time, e_frame=end_time)
    density_path = plot_norm_data(sub, run, chunk_num, chunked_df, sigma=sigma, output_path=output_gaze_maps)
    tmp_vector = flatten_img(density_path, bins=40)
    corr_container.append(tmp_vector)

    chunk_num +=1            
    sub_df = combine_data(chunk_list[int(run)], corr_container)