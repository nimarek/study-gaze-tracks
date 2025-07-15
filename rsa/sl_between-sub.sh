#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for chunk-$1 run-$2"
python3 /home/data/study_gaze_tracks/code/rsa/sl_between-sub.py $1 $2
