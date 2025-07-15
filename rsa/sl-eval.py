import os
import sys
import glob

import numpy as np
import nibabel as nib
from rsatoolbox.rdm.rdms import RDMs, load_rdm
from rsatoolbox.util.searchlight import evaluate_models_searchlight
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_fixed

import matplotlib.pyplot as plt
import seaborn as sns

# I/O
sub, run = str(sys.argv[1]), str(sys.argv[2])
distance, radius = "sqeuclidean", 2
model_dir, scratch_dir = "/home/data/study_gaze_tracks/derivatives/model_rdms/subj_wise", "/home/data/study_gaze_tracks/scratch"

# model evaluation
eval_metric = "spearman"
hypothesis = "ts2fix_50"

print(f"working on sub-{sub}, run-{run} and hypothesis:\t {hypothesis}")
sl_rdm_dir = scratch_dir + f"/scene-onset_analysis/neural-rdms_between-scenes"
save_folder = scratch_dir + f"/scene-onset_analysis/corr-maps_hypothesis-{hypothesis}"

filename = f"/sub-{sub}_run-{run}_hypothesis-{hypothesis}.nii.gz"
if os.path.isfile(save_folder + filename):
    raise ValueError(f"file already exists:\t {save_folder + filename}")

filename_hdf = f"/sub-{sub}_run-{run}_task-studyforrest"
rdm = load_rdm(sl_rdm_dir + filename_hdf, file_type="hdf5")
print("shape of target matrix:\t", rdm.dissimilarities[0].shape)

def load_candidates(candidate_dir, sub, run, hypothesis):

    models_rdm_list = []

    # print(candidate_dir + f"/sub-{sub}/sub-{sub}_run-{int(run)}_matrix.tsv")
    # for model in glob.glob(candidate_dir + f"/sub-{sub}/sub-{sub}_run-{int(run)}_matrix.tsv"):
    #     print("load general model:\t", model)
    #     eye_model = np.loadtxt(model, delimiter=",")
    #     models_rdm_list.append(eye_model)
    
    print(candidate_dir + f"/{hypothesis}/sub-{sub}_ses-movie_task-movie_run-0{int(run)}_rdm-ts2fix_dist-sqeuclidean.npy")
    for model in glob.glob(candidate_dir + f"/{hypothesis}/sub-{sub}_ses-movie_task-movie_run-0{int(run)}_rdm-ts2fix_dist-sqeuclidean.npy"):
        print("load general model:\t", model)
        eye_model = np.load(model)
        models_rdm_list.append(eye_model)
        
    # print(candidate_dir + f"/{hypothesis}/sub-{sub}_ses-movie_task-movie_run-0{int(run)}_rdm-sqeuclidean.npy")
    # for model in glob.glob(candidate_dir + f"/{hypothesis}/sub-{sub}_ses-movie_task-movie_run-0{int(run)}_rdm-sqeuclidean.npy"):
    #     print("load general model:\t", model)
    #     eye_model = np.load(model)
    #     models_rdm_list.append(eye_model)
    
    print(eye_model)
        
    return RDMs(
            dissimilarities=np.array(models_rdm_list),
            dissimilarity_measure="euclidean"
        )

def fisher_r_to_z(x):
    """    
    correct any rounding errors
    correlations cannot be greater than 1.
    """
    x = np.clip(x, -1, 1)
    return np.arctanh(x)

def plot_evaluation(sub, run, dataframe, save_folder):
    sns.distplot(dataframe)
    plt.title("Distributions of Correlations", size=18)
    plt.ylabel("Occurance", size=15)
    plt.xlabel("Spearmann Correlation", size=15)
    sns.despine()

    plot_name = save_folder + f"/sub-{sub}_run-{run}.png"

    plt.savefig(plot_name)
    plt.close()
    return None

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

weighted_model = ModelFixed("fixed model", load_candidates(
    model_dir, 
    sub, 
    run, 
    hypothesis=hypothesis
    )
)

# evaluate each voxel RDM with a fixed effects model
eval_results = evaluate_models_searchlight(
    sl_RDM=rdm,
    models=weighted_model,
    eval_function=eval_fixed,
    method=eval_metric,
    n_jobs=2)

eval_score = [np.float(e.evaluations) for e in eval_results]
# eval_score = fisher_r_to_z(eval_score)

# load mask
mask_path = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_mni/sub-{sub}/ses-movie/anat/sub-{sub}_ses-movie_label-GM_probseg_binary_mask.nii.gz"
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

RDM_brain = np.zeros([x * y * z])
RDM_brain[list(rdm.rdm_descriptors["voxel_index"])] = list(eval_score)
RDM_brain = RDM_brain.reshape([x, y, z])

if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

plot_evaluation(sub, run, eval_score, save_folder)

# save image
print(f"saving file to: {save_folder + filename} ...")
np2nii(mask_img, RDM_brain, save_folder + filename)
