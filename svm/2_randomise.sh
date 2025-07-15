#!/bin/bash

threshold=0.95
permutations=1000
hypothesis=${1}
output_path="/home/data/study_gaze_tracks/scratch/attention-mode_svm/lin-svm_hypothesis-${hypothesis}"

echo starting group-lvl for hypothesis-${hypothesis}

echo creating folders ...
mkdir -p ${output_path}

sub_id=1
sigma=2.354 # 2.354 equals to FWHM of 5

echo apply smoothing kernel ...
for image in /home/data/study_gaze_tracks/scratch/attention-mode_svm/*/*${hypothesis}*.nii.gz; do
    echo starting with image-${sub_id}
    # REMINDER: in fsl smoothing sigma is used instead of FWHM: FWHM = sigma*sqrt(8*ln(2)) = sigma*2.354
    fslmaths ${image} -s ${sigma} -sub 0.5 ${output_path}/img_${sub_id}_${hypothesis}_smoothed.nii.gz # 
    let "sub_id++"
done

echo merging input files ...
fslmerge -t ${output_path}/output_complete `ls ${output_path}/*${hypothesis}*_smoothed.nii.gz`

# echo calculating negative correlation values ...
# fslmaths ${output_path}/output_complete -mul -1 ${output_path}/output_complete

echo starting permutation-test ...
randomise -i ${output_path}/output_complete -o ${output_path}/perm_hypothesis-${hypothesis} -1 -T -v 6 -n ${permutations} -m /home/data/study_gaze_tracks/scratch/studf_avg152T1_gray_prob50_bin_resampled.nii

echo combining maps ...
fslmaths ${output_path}/perm_hypothesis-${hypothesis}_tfce_corrp_tstat1.nii.gz -thr ${threshold} -bin -mul ${output_path}/perm_hypothesis-${hypothesis}_tstat1.nii.gz ${output_path}/results-${hypothesis}_output_complete

echo extracting cluster information ...
cluster --in=${output_path}/results-${hypothesis}_output_complete --thresh=0.0001 --oindex=${output_path}/results-${hypothesis}_output_complete_cluster_index --olmax=${output_path}/results-${hypothesis}_output_complete_lmax.txt --osize=${output_path}/results-${hypothesis}_output_complete_cluster_size

echo removing marged input file
rm -r ${output_path}/*${hypothesis}*_smoothed.nii.gz
rm ${output_path}/output_complete.nii.gz