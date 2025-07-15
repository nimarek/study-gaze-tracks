#!/bin/bash

source /home/data/software/experimental/ipsy-env/activate
echo "calling python script with for sub-$1 and run-$2"

python run-lss_roi.py --sub=$1 --run=$2 --task=movie \
    --space=T1w --fwhm=3. \
    --model_type=lss \
    --bidsroot=/home/data/study_gaze_tracks/studyforrest-data-phase2/ \
    --fmriprep_dir=/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_native/