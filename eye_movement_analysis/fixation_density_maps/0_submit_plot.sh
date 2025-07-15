#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_spatial_plots/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 32MB
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/spatial-attention_analysis/0_eye-maps
executable     = 0_plot_fdm.sh
\n"

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do # 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36
    for run in {1..8}; do
        printf "arguments = ${sub} ${run}\n"
        printf "log       = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).log\n"
        printf "output    = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).out\n"
        printf "error     = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).err\n"
        printf "Queue\n\n"
    done
done
