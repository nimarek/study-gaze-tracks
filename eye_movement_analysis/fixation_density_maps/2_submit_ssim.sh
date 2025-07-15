#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_ssim/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 2GB
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/spatial-attention_analysis/0_eye-maps
executable     = 2_struct-sim.sh
\n"

for run in {1..8}; do
    printf "arguments = ${run}\n"
    printf "log       = ${logs_dir}/run-${run}_\$(Cluster).\$(Process).log\n"
    printf "output    = ${logs_dir}/run-${run}_\$(Cluster).\$(Process).out\n"
    printf "error     = ${logs_dir}/run-${run}_\$(Cluster).\$(Process).err\n"
    printf "Queue\n\n"
done
