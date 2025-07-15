#!/bin/bash

source /home/data/software/experimental/ipsy-env/activate
echo "calling fmriprep container for sub-$1"

#User inputs:
bids_root_dir=/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_native
tmp_work_dir=/home/nico/scratch

#Run fmriprep
mriqc $bids_root_dir $bids_root_dir/derivatives/mriqc \
  participant \
  --participant-label $1 \
  --nprocs 4 \
  --mem 8 \
  --verbose-reports \
  -w $tmp_work_dir