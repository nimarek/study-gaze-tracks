import os
import re
import sys
import glob

import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    get_searchlight_RDMs,
)
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.util.searchlight import evaluate_models_searchlight
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_fixed

chunk, run = str(sys.argv[1]), str(sys.argv[2])

# analysis hyperparameters
radius = 2
distance = "euclidean"
hypothesis = "spatial-attention"

mni_mask = datasets.load_mni152_brain_mask()
print("original MNI mask:", mni_mask)

ref_img_path = "/home/data/study_gaze_tracks/scratch/lss_scene-onset-beta/sub-01/sub-01_run-1_contrast-chunk1_beta-map.nii.gz"
ref_img = nib.load(ref_img_path)
print("reference image loaded:", ref_img.shape)

mask_img = resample_to_img(mni_mask, ref_img, interpolation="nearest")

# paths and stuff
print(f"working on chunk-{chunk} in run-{run} ...")
deriv_dir, scratch_dir = "/home/data/study_gaze_tracks/derivatives", "/home/data/study_gaze_tracks/scratch"
beta_dir = scratch_dir + f"/lss_scene-onset-beta/sub-*"
save_folder = scratch_dir + f"/scene-onset_analysis/corr-maps_hypothesis-fix_cnt_between-sub"
filename = f"/chunk-{chunk}_run-{run}_hypothesis-{hypothesis}.nii.gz"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def set_paths(chunk=None, run=None, beta_dir=None):
    """
    Function to create paths pointing to appropriate files for between- 
    or within-chunkject analysis. Defaults to between-chunkject analysis.
    Call function once for within and n*chunk times for between-chunkject
    analysis.
    """
    image_paths = glob.glob(beta_dir + f"/*_run-{int(run)}_contrast-chunk{chunk}_beta-map.nii.gz")
    image_paths.sort(key=natural_keys)
    return image_paths

def upper_tri(RDM):
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def create_dataset(image_paths, ref_img):
    labels = []

    ref_img_data = ref_img.get_fdata()
    x, y, z = ref_img_data.shape

    # create dummy df according to coordinates
    data = np.zeros((len(image_paths), x, y, z))

    for x, im in enumerate(image_paths):
        print("contrast loaded:\t", im.split("/")[-1])
        labels.append(im.split("/")[-1])
        data[x] = nib.load(im).get_fdata()
    return data, labels

def fisher_r_to_z(x):
    """    
    correct any rounding errors
    correlations cannot be greater than 1.
    """
    x = np.clip(x, -1, 1)
    return np.arctanh(x)

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

# load mask
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

# create dataset
image_paths = set_paths(chunk=chunk, run=run, beta_dir=beta_dir)
data, labels = create_dataset(image_paths, mask_img) 

# reshape data so we have n_observastions x n_voxels
data_2d = data.reshape([data.shape[0], -1])
mask_2d = ~np.all(data_2d==0, axis=0)
mask_3d = mask_2d.reshape(x, y, z)

print("shape of 2d data ...\t", data_2d.shape)

centers, neighbors = get_volume_searchlight(mask_3d, radius=radius, threshold=0.)
neural_rdm = get_searchlight_RDMs(data_2d, centers, neighbors, labels, method=distance)

# create list to load mult. models
models_rdm_list = []

# load temporal model
model_fpath = os.path.join(deriv_dir, "model_rdms", "chunk_wise", "fix_cnt", "sqeuclidean", f"ses-movie_task-movie_run-{run}_chunk{chunk}_rdm-fixcnt_dist-diff.npy")

print("load general model:\t", model_fpath)
temp_model = upper_tri(np.load(model_fpath))
models_rdm_list.append(temp_model)

candidate_model = ModelFixed("fixed model",
                             RDMs(
                                 dissimilarities=np.array(models_rdm_list),
                                 dissimilarity_measure="euclidean"
                                 )
)

# evaluate each voxel RDM with a fixed effects model
eval_results = evaluate_models_searchlight(
    sl_RDM=neural_rdm,
    models=candidate_model,
    eval_function=eval_fixed,
    method="spearman",
    n_jobs=2)

eval_score = [np.float(e.evaluations) for e in eval_results]
# eval_score = fisher_r_to_z(eval_score)

RDM_brain = np.zeros([x * y * z])
RDM_brain[list(neural_rdm.rdm_descriptors["voxel_index"])] = list(eval_score)
RDM_brain = RDM_brain.reshape([x, y, z])

if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

# save image
print(f"saving file to: {save_folder + filename} ...")
np2nii(mask_img, RDM_brain, save_folder + filename)
