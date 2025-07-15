#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for soi-$1 run-$2"
python3 /home/data/study_gaze_tracks/code/spatial-attention_analysis/0_eye-maps/0_plot-maps.py $1 $2 