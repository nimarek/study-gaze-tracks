#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for sub-$1"
python3 /home/data/study_gaze_tracks/code/spatial-attention_analysis/1_svm/1_lin_svm.py $1