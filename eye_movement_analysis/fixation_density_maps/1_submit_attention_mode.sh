#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_attention_mode/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 32MB
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/spatial-attention_analysis/0_eye-maps
executable     = 1_attention-mode.sh
\n"

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
    for sub_b in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
        for run in {1..8}; do
            printf "arguments = ${sub} ${sub_b} ${run}\n"
            printf "log       = ${logs_dir}/sub-${sub}_sub-${sub_b}_run-${run}\$(Cluster).\$(Process).log\n"
            printf "output    = ${logs_dir}/sub-${sub}_sub-${sub_b}_run-${run}\$(Cluster).\$(Process).out\n"
            printf "error     = ${logs_dir}/sub-${sub}_sub-${sub_b}_run-${run}\$(Cluster).\$(Process).err\n"
            printf "Queue\n\n"
        done
    done
done
