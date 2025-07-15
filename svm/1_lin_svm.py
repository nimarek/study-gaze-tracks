import re
import os
import glob
import sys

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold # LeaveOneGroupOut 
from sklearn.svm import LinearSVC

from nilearn.image import concat_imgs
from nibabel import save
from nilearn.decoding import SearchLight
from nilearn.image import load_img

sub = str(sys.argv[1])
sl_radius = 4

scoring = "accuracy" 
print(f"using scoring metric:", scoring)

out_dir = "/home/data/study_gaze_tracks/scratch/attention-mode_svm"
# betas_dir = f"/home/data/study_gaze_tracks/scratch/lss_spatial-attention_scene-onset_t1w/run-*"
betas_dir = f"/home/data/study_gaze_tracks/scratch/lss_spatial-attention_scene-onset_mni-t/run-*"
label_f = f"/home/data/study_gaze_tracks/derivatives/euclidean-distance-models/attention-mode_cluster-analysis/sub-{sub}/sub-{sub}_combined-labels.tsv"

# set paths to store output
output_folder = out_dir + f"/sub-{sub}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# set file name
filename =  output_folder + f"/sub-{sub}_attention-mode.nii.gz"
if os.path.isfile(output_folder + filename):
    raise Exception("file already exists, abort mission ...")

# define usefull functions
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]

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
    img = nib.load(img)
    header = nib.Nifti1Header()
    affine = img.affine
    nii_file = nib.Nifti1Image(scores, affine, header)
    nib.save(nii_file, filename)
    return nii_file

def prepare_data(input_func_dir, sub, label_f):
    combined_global = []
    classi_inf = pd.read_csv(label_f, sep="\t")

    labels = classi_inf["label"]
    chunks = classi_inf["run"]

    print("label 0:", np.count_nonzero(labels == 0))
    print("label 1:", np.count_nonzero(labels == 1))

    print(f"loading data ...")

    func_runs = glob.glob(input_func_dir + f"/sub-{sub}_run-*_space-*_desc-chunk-*.nii.gz")
    func_runs.sort(key=natural_keys)

    """
    create df for labels and paths, delete paths
    according to cluster labels.
    """
    classi_inf["lss_paths"] = func_runs
    classi_inf = classi_inf.drop(classi_inf[classi_inf["label"] > 1].index)

    func_runs = classi_inf["lss_paths"]
    labels = classi_inf["label"]
    chunks = classi_inf["run"]

    # END
    combined_img = list(map(load_img, func_runs))
    combined_global.append(combined_img)
    
    combined_global = concat_imgs(combined_global)
    labels_global = np.hstack(labels)
    chunks_global = np.hstack(chunks)

    print("labels:\t", labels_global.shape)
    print("chunks:\t", chunks_global.shape)
    print("shape of combined signal:\t", combined_global.shape)
    return combined_global, labels_global, chunks_global

cv = KFold(n_splits=5) # LeaveOneGroupOut()

decoder = Pipeline([
        ("scale", StandardScaler()),
        ("svm", LinearSVC(penalty="l2", dual=True, class_weight="balanced", loss="squared_hinge", max_iter=2000)) 
])

brain_mask_path = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep/sub-{sub}/ses-movie/func/sub-{sub}_ses-movie_task-movie_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
# brain_mask_path = "/home/data/study_gaze_tracks/scratch/studf_avg152T1_gray_prob50_bin_resampled.nii"
brain_mask = load_img(brain_mask_path)
combined_global, labels, chunks = prepare_data(input_func_dir=betas_dir, sub=sub, label_f=label_f)

if not combined_global.shape[3] == labels.shape[0] == chunks.shape[0]:
    raise ValueError("Dimensions do not add up!")

sl = SearchLight(
    mask_img=brain_mask,
    radius=sl_radius,
    estimator=decoder,
    n_jobs=4,
    scoring=scoring,
    cv=cv,
    verbose=False)

print("starting to fit individual searchlights ...")
sl.fit(imgs=combined_global, y=labels, groups=chunks)

print("saving img ...", filename)
np2nii(brain_mask_path, sl.scores_, filename)