logs_dir=/home/data/study_gaze_tracks/code/logs_roi-lss/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# exclude bad nodes from analysis

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 20G
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/lss
executable     = roi.sh
\n"

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
    for run in 1; do
        printf "arguments = ${sub} ${run}\n"
        printf "log       = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).log\n"
        printf "output    = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).out\n"
        printf "error     = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).err\n"
        printf "Queue\n\n"
    done
done