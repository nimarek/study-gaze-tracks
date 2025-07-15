import os
import nibabel as nib
from nilearn.image import binarize_img, math_img, resample_to_img

sub_list = ["01", "02", "03", "04", "05", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]
prob = 0.2 

def create_gm_mask(mask_path, func_ref_path, prob=0.5):
    """
    Create a binary gray matter mask from a probability map.

    Parameters:
    - mask_path: Path to the gray matter probability map
    - func_ref_path: Path to the functional reference image
    - prob: Probability threshold

    Returns:
    - A resampled binary gray matter mask
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Gray matter probability map not found: {mask_path}")
    if not os.path.exists(func_ref_path):
        raise FileNotFoundError(f"Functional reference image not found: {func_ref_path}")

    gm_binary = binarize_img(math_img(f"img >= {prob}", img=mask_path), threshold=prob)
    return resample_to_img(gm_binary, func_ref_path, interpolation="nearest")
   
for sub in sub_list:
    try:
        print(f"Processing subject-{sub}...")
        anat_dir = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_mni/sub-{sub}/ses-movie/anat"
        func_dir = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_mni/sub-{sub}/ses-movie/func"
        # gm_probseg_path = os.path.join(anat_dir, f"sub-{sub}_ses-movie_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz")
        # func_ref_path = os.path.join(func_dir, f"sub-{sub}_ses-movie_task-movie_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
        
        gm_probseg_path = os.path.join(anat_dir, f"sub-{sub}_ses-movie_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz")
        func_ref_path = os.path.join(func_dir, f"sub-{sub}_ses-movie_task-movie_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
        
        gm_mask = create_gm_mask(mask_path=gm_probseg_path, func_ref_path=func_ref_path, prob=prob)

        output_path = os.path.join(anat_dir, f"sub-{sub}_ses-movie_label-GM_probseg_binary_mask.nii.gz")
        nib.save(gm_mask, output_path)
        print(f"Saved gray matter mask for subject-{sub} to {output_path}")

    except Exception as e:
        print(f"Error processing subject-{sub}: {e}")
