#!/bin/bash

threshold=0.95
permutations=100
hypothesis=${1}
output_path="/home/data/study_gaze_tracks/scratch/sqeuclidean-distance-models_scene-onset/${hypothesis}_radius-2"

echo starting group-lvl for hypothesis-${hypothesis}

echo creating folders ...
mkdir -p ${output_path}

echo merging input files ...
fslmerge -t ${output_path}/output_complete `ls ${output_path}/*${hypothesis}*.nii.gz`

# echo calculating negative correlation values ...
# fslmaths ${output_path}/output_complete -mul -1 ${output_path}/output_complete

echo starting permutation-test ...
randomise -i ${output_path}/output_complete -o ${output_path}/perm_hypothesis-${hypothesis} -1 -T -v 4 -n ${permutations} # -m /home/data/study_gaze_tracks/scratch/studf_avg152T1_gray_prob50_bin_resampled.nii

echo combining maps ...
fslmaths ${output_path}/perm_hypothesis-${hypothesis}_tfce_corrp_tstat1.nii.gz -thr ${threshold} -bin -mul ${output_path}/perm_hypothesis-${hypothesis}_tstat1.nii.gz ${output_path}/results-${hypothesis}_output_complete

echo extracting cluster information ...
cluster --in=${output_path}/results-${hypothesis}_output_complete --thresh=0.0001 --oindex=${output_path}/results-${hypothesis}_output_complete_cluster_index --olmax=${output_path}/results-${hypothesis}_output_complete_lmax.txt --osize=${output_path}/results-${hypothesis}_output_complete_cluster_size

echo removing marged input file
rm -r ${output_path}/*${hypothesis}*_smoothed.nii.gz
rm ${output_path}/output_complete.nii.gz
