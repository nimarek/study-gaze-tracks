import re
import os
import glob
import sys
import re

import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from nilearn.image import concat_imgs
from nilearn.decoding import SearchLight
from nilearn.image import load_img

sub = str(sys.argv[1])
sl_radius = 2

scoring = "accuracy" 
print(f"using scoring metric:", scoring)

out_dir = "/home/exp-psy/Desktop/study_face_tracks/derivatives/face-vs-no-face_svm"
betas_dir = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/lss_face-no-face-beta/sub-{sub}"

# set paths to store output
output_folder = out_dir + f"/sub-{sub}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# set file name
filename =  output_folder + f"/sub-{sub}_face-vs-no-face.nii.gz"
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

def prepare_data(input_func_dir, sub):
    """
    Load and preprocess subject-specific beta images for face vs. no-face classification.

    Parameters
    ----------
    input_func_dir : str
        Directory containing beta images for the subject.
    sub : str
        Subject ID (e.g., "01").

    Returns
    -------
    combined_imgs : Nifti1Image
        4D Nifti image with all beta maps concatenated.
    labels : ndarray
        1D array with binary labels (1 = face, 0 = no-face).
    chunks : ndarray
        1D array with chunk labels (e.g., run numbers).
    """

    print("Loading beta maps for subject:", sub)
    beta_paths = glob.glob(
        os.path.join(input_func_dir, f"sub-{sub}_run-*_contrast-*_beta-map.nii.gz")
    )
    beta_paths.sort(key=natural_keys)

    if len(beta_paths) == 0:
        raise FileNotFoundError(f"No beta files found for sub-{sub} in {input_func_dir}")

    labels = []
    chunks = []
    imgs = []

    for path in beta_paths:
        filename = os.path.basename(path)

        # binary label
        if "contrast-face" in filename:
            label = 1
        elif "contrast-noface" in filename or "contrast-no-face" in filename:
            label = 0
        else:
            print(f"Skipping unrecognized contrast: {filename}")
            continue

        # run numbers
        run_match = re.search(r"run-(\d+)", filename)
        if run_match:
            run = int(run_match.group(1))
        else:
            raise ValueError(f"could not extract run number from: {filename} ...")

        labels.append(label)
        chunks.append(run)
        imgs.append(load_img(path))

    combined_imgs = concat_imgs(imgs)
    labels = np.array(labels, dtype=int)
    chunks = np.array(chunks, dtype=int)

    print(f"Loaded {len(labels)} beta maps:")
    print(f" - Faces:     {np.sum(labels == 1)}")
    print(f" - No-Faces:  {np.sum(labels == 0)}")
    print(f" - Chunks:    {np.unique(chunks)}")
    print(f" - Image shape: {combined_imgs.shape}")
    return combined_imgs, labels, chunks

cv = StratifiedKFold(n_splits=5)

decoder = Pipeline([
        ("scale", StandardScaler()),
        ("svm", LinearSVC(penalty="l2", dual=True, class_weight="balanced", loss="squared_hinge", max_iter=2000)) 
])

brain_mask_path = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/fmriprep_mni/sub-{sub}/ses-movie/anat/sub-{sub}_ses-movie_label-GM_probseg_binary_mask.nii.gz"
brain_mask = load_img(brain_mask_path)
combined_global, labels, chunks = prepare_data(input_func_dir=betas_dir, sub=sub)

if not combined_global.shape[3] == labels.shape[0] == chunks.shape[0]:
    raise ValueError("Dimensions do not add up!")

sl = SearchLight(
    mask_img=brain_mask,
    radius=sl_radius,
    estimator=decoder,
    n_jobs=-1,
    scoring=scoring,
    cv=cv,
    verbose=False)

print("starting to fit individual searchlights ...")
sl.fit(imgs=combined_global, y=labels, groups=chunks)

print("saving img ...", filename)
np2nii(brain_mask_path, sl.scores_, filename)