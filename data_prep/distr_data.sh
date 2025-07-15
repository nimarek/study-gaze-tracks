#!/bin/bash

src_dir="/home/data/study_gaze_tracks/derivatives/reference_spatial-attention"
src_pattern="ses-movie_task-movie_run-0*_events.tsv"
dest_base="/home/data/study_gaze_tracks/studyforrest-data-phase2"

for subj_path in ${dest_base}/sub-*/ses-movie/func/; do
    subj_id=$(basename $(dirname $(dirname "$subj_path")))
    run_idx=1
    for src_file in ${src_dir}/${src_pattern}; do
        dest_file="${subj_path}/${subj_id}_ses-movie_task-movie_run-${run_idx}_events.tsv"
        cp "$src_file" "$dest_file"
        echo "Copied to $dest_file"
        ((run_idx++))
    done
done
