logs_dir=/home/data/study_gaze_tracks/code/logs_hyperalignment/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 8
request_memory = 32000

# Execution
initial_dir    = /home/data/study_gaze_tracks/code/0_fmri-scripts
executable     = 4_hyperalignment.py
\n"

# # prepare datasets
# for sub in 0{1..9} {10..20}; do
#         printf "arguments = ${sub}\n"
#         printf "log       = ${logs_dir}/sub-${sub}_ha-\$(Cluster).\$(Process).log\n"
#         printf "output    = ${logs_dir}/sub-${sub}_ha-\$(Cluster).\$(Process).out\n"
#         printf "error     = ${logs_dir}/sub-${sub}_ha-\$(Cluster).\$(Process).err\n"
#         printf "Queue\n\n"
#     done
# done

# run actual hyperalignment 
for sl_radius in {5..8}; do
    printf "arguments = ${sl_radius}\n"
    printf "log       = ${logs_dir}/combined_sl-${sl_radius}-\$(Cluster).\$(Process).log\n"
    printf "output    = ${logs_dir}/combined_sl-${sl_radius}-\$(Cluster).\$(Process).out\n"
    printf "error     = ${logs_dir}/combined_sl-${sl_radius}-\$(Cluster).\$(Process).err\n"
    printf "Queue\n\n"
done