#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for soi-$1 and sub-comp-$2 run-$3"
python3 /home/data/study_gaze_tracks/code/spatial-attention_analysis/0_eye-maps/1_attention-mode.py $1 $2 $3