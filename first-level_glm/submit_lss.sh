logs_dir=/home/data/study_gaze_tracks/code/logs_lss/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# exclude bad nodes from analysis

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 4
request_memory = 5G
request_disk = 5G

# Execution
initial_dir    = /home/data/study_gaze_tracks/code/nilearn-lss
executable     = lss.sh
\n"

for sub in 01 02 03 04 06 09 10 14 15 16 17 18 19 20; do
    for run in {1..8}; do
        printf "arguments = ${sub} ${run}\n"
        printf "log       = ${logs_dir}/sub-${sub}_run-${run}_\$(Cluster).\$(Process).log\n"
        printf "output    = ${logs_dir}/sub-${sub}_run-${run}_\$(Cluster).\$(Process).out\n"
        printf "error     = ${logs_dir}/sub-${sub}_run-${run}_\$(Cluster).\$(Process).err\n"
        printf "Queue\n\n"
    done
done
