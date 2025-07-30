import os
import glob
import sys

import numpy as np
import nibabel as nib
from rsatoolbox.rdm.rdms import load_rdm
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import evaluate_models_searchlight

sub = str(sys.argv[1])

base_dir = "/home/exp-psy/Desktop/study_face_tracks"
deriv_dir = os.path.join(base_dir, "derivatives")
in_dir = "/home/exp-psy/Desktop/study_face_tracks/scratch/perc-valence_analysis/neural-rdms_between-scenes" 
out_dir = "/home/exp-psy/Desktop/study_face_tracks/scratch/perc-valence_analysis"

# save image
sub_folder_save = out_dir + f"/corr-maps_perc-valence/sub-{sub}"
if not os.path.exists(sub_folder_save):
    os.makedirs(sub_folder_save, exist_ok=True)
filename_save = f"/sub-{sub}_task-movie_hypothesis-perc-valence.nii.gz"

if os.path.isfile(sub_folder_save + filename_save):
    raise Exception("file already exists, abort mission ...")

def upper_tri(RDM):
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def np2nii(img, scores, filename):
    """
    It saves data into a nifti file
    by leveraging on another image
    of the same size.
    Parameters
    ----------
    img : nifti file (e.g a mask)
    scores : numpy array containing decoding scores
    filename : string name of the new nifti file
    Returns
    -------
    nii_file : Nifti1Image
    """
    header = nib.Nifti1Header()
    affine = img.affine
    nii_file = nib.Nifti1Image(scores, affine, header)
    nib.save(nii_file, filename)
    return nii_file

def fisher_r_to_z(x):
    """    
    correct any rounding errors
    correlations cannot be greater than 1.
    """
    x = np.clip(x, -1, 1)
    return np.arctanh(x)

# load model rdm containing specific hypothesis about relationships
perc_valence_matrix = np.load(os.path.join(base_dir, "code", "flat_crossval_distances.npy"))
eval_model = ModelFixed("Valence RDM", perc_valence_matrix)

# load sl rdms for each voxel
filename_hdf = f"/sub-{sub}_task-studyforrest"
rdm = load_rdm(in_dir + filename_hdf, file_type="hdf5")

print(f"working on sub-{sub}. Shape of the model rdm: {perc_valence_matrix.shape}, shape of the neural rdm: {rdm.dissimilarities.shape}")

# evaluate each voxel RDM with a fixed effects model
eval_results = evaluate_models_searchlight(
    sl_RDM=rdm,
    models=eval_model,
    eval_function=eval_fixed,
    method="corr_cov",
    n_jobs=4,
)
eval_score = [float(e.evaluations) for e in eval_results]
# eval_score_z = fisher_r_to_z(eval_score)

# load mask
mask_path = os.path.join(
    deriv_dir, f"fmriprep_mni/sub-{sub}/ses-movie/anat",
    f"sub-{sub}_ses-movie_label-GM_probseg_binary_mask.nii.gz"
)
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

RDM_brain = np.zeros([x * y * z])
RDM_brain[list(rdm.rdm_descriptors["voxel_index"])] = list(eval_score)
RDM_brain = RDM_brain.reshape([x, y, z])

print(f"saving file to: {sub_folder_save + filename_save} ...")
np2nii(mask_img, RDM_brain, sub_folder_save + filename_save)