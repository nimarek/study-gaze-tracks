#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for sub-$1 run-$2"
python3 /home/data/study_gaze_tracks/code/0_eye-maps/1_pupil_dil.py $1 $2