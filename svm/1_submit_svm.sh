#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_lin-svm/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 4
request_memory = 3G
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/spatial-attention_analysis/1_svm/
executable     = 1_svm.sh
\n"

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
    printf "arguments = ${sub}\n"
    printf "log       = ${logs_dir}/sub-${sub}_\$(Cluster).\$(Process).log\n"
    printf "output    = ${logs_dir}/sub-${sub}_\$(Cluster).\$(Process).out\n"
    printf "error     = ${logs_dir}/sub-${sub}_\$(Cluster).\$(Process).err\n"
    printf "Queue\n\n"
done