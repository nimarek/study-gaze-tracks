#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for run-$1"
python3 /home/data/study_gaze_tracks/code/spatial-attention_analysis/0_eye-maps/2_ssim.py $1