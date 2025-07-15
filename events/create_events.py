import numpy as np
import pandas as pd

sub_list = ["01", "02", "03", "04", "05", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]
run_list = ["1", "2", "3", "4", "5", "6", "7", "8"]

def load_events(annotation_dir, run, complete_analysis=False):
    """
    TR is 2 seconds, add 1 to onsets because of slice timing correction. Check
    https://fmriprep.org/en/stable/outputs.html for more infos.
    """
    # load for scene onset analysis
    tmp = pd.read_csv(annotation_dir + f"/ses-movie_task-movie_run-0{run}_events.tsv", sep="\t") 
    # tmp = pd.read_csv(annotation_dir + f"/studyf_run-0{run}_face-orientation.csv", sep=",") 
    tmp = tmp[["onset", "duration", "trial_type"]]
    tmp = tmp.assign(onset=tmp["onset"] - 1, duration=tmp["duration"] * 4)

    return tmp

def prolong_chunks(onset_list, duration=4.0):
    """
    Get onsets between location switches for spatial-attention
    RS-Analysis.
    """
    onset_container = []
    for start, stop in zip(onset_list[:-1], onset_list[1:]):
        onset_container.append(np.arange(start, stop, duration))
    return np.concatenate(onset_container).tolist()

for sub in sub_list:
    for run in run_list: 
        out_dir = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/sub-{sub}/ses-movie/func"
        out_name = out_dir + f"/sub-{sub}_ses-movie_task-movie_run-{run}_events.tsv"
        print("saving to ...\t", out_name)

        # load for scene onset analysis
        load_events(annotation_dir="/home/data/study_gaze_tracks/code/reference_spatial-attention", run=run, complete_analysis=True).to_csv(out_name, sep="\t", index=False) 