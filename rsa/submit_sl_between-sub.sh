#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_neural-rdm/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 4
request_memory = 10G
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/rsa/
executable     = sl_between-sub.sh
\n"

declare -A max_chunks=(
    [01]=59
    [02]=68
    [03]=55
    [04]=71
    [05]=54
    [06]=68
    [07]=83
    [08]=51
)

for run in "${!max_chunks[@]}"; do
    max_chunk=${max_chunks[$run]}
    for ((chunk=1; chunk<=max_chunk; chunk++)); do
        printf "arguments = %d %s\n" "$chunk" "$run"
        printf "log       = %s/chunk-%d_run-%s\$(Cluster).\$(Process).log\n" "$logs_dir" "$chunk" "$run"
        printf "output    = %s/chunk-%d_run-%s\$(Cluster).\$(Process).out\n" "$logs_dir" "$chunk" "$run"
        printf "error     = %s/chunk-%d_run-%s\$(Cluster).\$(Process).err\n" "$logs_dir" "$chunk" "$run"
        printf "Queue\n\n"
    done
done
