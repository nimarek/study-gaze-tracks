#!/bin/bash

source /home/data/software/experimental/ipsy-env/activate
echo "calling fmriprep container for sub-$1"

#User inputs:
bids_root_dir=/home/data/study_gaze_tracks/studyforrest-data-phase2
tmp_work_dir=/home/nico/scratch
output_space=MNI152NLin2009cAsym:res-2
nthreads=12
mem=32 #gb

mem=`echo "${mem//[!0-9]/}"` 
mem_mb=`echo $(((mem*1000)-5000))` 

#Run fmriprep
fmriprep $bids_root_dir $bids_root_dir/derivatives \
  participant \
  --participant-label $1 \
  --skip-bids-validation \
  --md-only-boilerplate \
  --fs-license-file $bids_root_dir/code/license/license.txt \
  --output-spaces $output_space \
  --nthreads $nthreads \
  --stop-on-first-crash \
  --mem_mb $mem_mb \
  -w $tmp_work_dir