import re
import os
import glob
import sys
import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from nilearn.image import concat_imgs, load_img
from nilearn.masking import apply_mask

sub = str(sys.argv[1])
hemi = str(sys.argv[2])
sl_radius = 2
scoring = "accuracy"

print("Using scoring metric:", scoring)
print("Using hemisphere:", hemi)

# Paths
prep_dir = "/home/exp-psy/Desktop/study_face_tracks/derivatives/fmriprep_mni"
out_dir = "/home/exp-psy/Desktop/study_face_tracks/derivatives/face-vs-no-face_roi"
betas_dir = f"/home/exp-psy/Desktop/study_face_tracks/derivatives/lss_face-no-face-beta/sub-{sub}"

output_folder = os.path.join(out_dir, f"sub-{sub}")
os.makedirs(output_folder, exist_ok=True)

filename = os.path.join(output_folder, f"sub-{sub}_face-vs-no-face_roi-{hemi}.txt")
if os.path.isfile(filename):
    raise Exception("File already exists. Abort mission ...")

def build_mask(prep_dir, subject_id, hemi="both", left_roi="ctx-lh-fusiform", right_roi="ctx-rh-fusiform"):
    # load segmentation df
    lut_df = pd.read_csv(os.path.join(prep_dir, "desc-aparcaseg_dseg.tsv"), sep="\t")

    indices = []
    if hemi in ["left", "both"]:
        left_index = lut_df[lut_df["name"] == left_roi]["index"].values[0]
        indices.append(left_index)
    if hemi in ["right", "both"]:
        right_index = lut_df[lut_df["name"] == right_roi]["index"].values[0]
        indices.append(right_index)

    aparc_fpath = os.path.join(
        prep_dir, f"sub-{subject_id}", "ses-movie", "func",
        f"sub-{subject_id}_ses-movie_task-movie_run-1_space-MNI152NLin2009cAsym_res-2_desc-aparcaseg_dseg.nii.gz"
        )
    
    aparc_img = nib.load(aparc_fpath)
    aparc_data = aparc_img.get_fdata()

    mask_data = np.isin(aparc_data, indices)
    return nib.Nifti1Image(mask_data.astype(np.float32), affine=aparc_img.affine)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]

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


betas, labels, chunks = prepare_data(betas_dir, sub)
roi_mask = build_mask(prep_dir, subject_id=sub, hemi=hemi)


X = apply_mask(betas, roi_mask)

# cv = StratifiedKFold(n_splits=5)
cv = GroupKFold(n_splits=5)
clf = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("svm", LinearSVC(penalty="l2", dual=True, class_weight="balanced", loss="squared_hinge", max_iter=2000, random_state=42))
])

print("Starting ROI-based classification ...")
# scores = cross_val_score(clf, X, labels, cv=cv, scoring=scoring, n_jobs=-1)

balanced_acc = make_scorer(balanced_accuracy_score)
scores = cross_val_score(clf, X, labels, groups=chunks, cv=cv, scoring=balanced_acc)

mean_score = np.mean(scores)

print("Classification accuracy:", mean_score)

# # Save result
# with open(filename, "w") as f:
#     f.write(f"{mean_score:.4f}\n")
