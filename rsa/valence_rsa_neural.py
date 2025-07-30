import os
import re
import sys
import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
from nilearn.image import smooth_img, load_img, get_data
from rsatoolbox.data.noise import prec_from_residuals, prec_from_measurements
from rsatoolbox.util.searchlight import get_volume_searchlight
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm import RDMs

sub = str(sys.argv[1])
fwhm, radius = 1, 2
distance = "crossnobis"

base_dir = "/home/exp-psy/Desktop/study_face_tracks"
deriv_dir = os.path.join(base_dir, "derivatives")
scratch_dir = os.path.join(base_dir, "scratch")
beta_dir = os.path.join(scratch_dir, f"lss_perc-valence-beta", f"sub-{sub}")

output_folder = os.path.join(scratch_dir, "perc-valence_analysis", "neural-rdms_between-scenes")
os.makedirs(output_folder, exist_ok=True)

filename = f"/sub-{sub}_task-studyforrest"
if os.path.isfile(output_folder + filename):
    raise FileExistsError("File already exists, aborting.")

print(f"running sub-{sub} with metric: {distance} and FWHM: {fwhm}")

def atoi(text): return int(text) if text.isdigit() else text
def natural_keys(text): return [atoi(c) for c in re.split(r"(\d+)", text)]

def get_image_paths_by_run(sub, beta_dir, residuals=False):
    image_paths, run_ids = [], []
    grouped_paths = []

    # use only runs with balanced trials
    for run in [1, 2, 3, 5, 6, 7]:
        tsv_path = os.path.join(deriv_dir, "reference_face-emotions", f"run-0{run}_lsa-valence.tsv")
        events_df = pd.read_csv(tsv_path, sep="\t")
        trial_types = np.unique(events_df["trial_type"])

        paths_this_run = []
        for trial in trial_types:
            suffix = "residuals" if residuals else "beta-map"
            pattern = f"sub-{sub}_run-{run}_contrast-{trial}_{suffix}.nii.gz"
            f_path = os.path.join(beta_dir, pattern)
            match = glob.glob(f_path)

            if not match:
                print(f"missing file: {trial} (Run {run})")
                continue

            paths_this_run.append(match[0])
            if not residuals:
                image_paths.append(match[0])
                run_ids.append(str(run))

        if residuals:
            grouped_paths.append(paths_this_run)

    return (image_paths, run_ids) if not residuals else grouped_paths

def create_dataset(image_paths, mask_path, fwhm):
    ref_img = load_img(mask_path)
    x, y, z = ref_img.shape
    data = np.zeros((len(image_paths), x, y, z))
    labels = []

    for i, path in enumerate(image_paths):
        label = os.path.basename(path).split("contrast-")[1].split("_")[0]
        labels.append(label)
        print("contrast loaded:\t", label)
        data[i] = smooth_img(load_img(path), fwhm).get_fdata()
    return data, labels

def create_residual_dataset(image_paths_by_run, mask_path, fwhm):
    mask_img = load_img(mask_path)
    mask_data = get_data(mask_img).astype(bool)
    
    data = []

    for run_idx, run_paths in enumerate(image_paths_by_run):
        run_data = []
        for trial_idx, path in enumerate(run_paths):
            img = load_img(path)
            smoothed_img = smooth_img(img, fwhm)
            trial_data = get_data(smoothed_img)[mask_data]  # 1D vector
            run_data.append(trial_data)

        run_array = np.stack(run_data, axis=0)
        data.append(run_array)

    residuals = np.stack(data, axis=0)
    return residuals

mask_path = os.path.join(
    deriv_dir, f"fmriprep_mni/sub-{sub}/ses-movie/anat",
    f"sub-{sub}_ses-movie_label-GM_probseg_binary_mask.nii.gz"
)
mask_data = load_img(mask_path).get_fdata()
x, y, z = mask_data.shape

image_paths, runs = get_image_paths_by_run(sub, beta_dir, residuals=False)
data, conditions = create_dataset(image_paths, mask_path, fwhm)
obs_descriptors = {"condition": conditions, "run": runs}
data_2d = np.nan_to_num(data.reshape(data.shape[0], -1))

centers, neighbors = get_volume_searchlight(mask_data, radius=radius, threshold=0.)

# # residuals
# image_paths_res = get_image_paths_by_run(sub, beta_dir, residuals=True)
# data_res = create_residual_dataset(image_paths_res, mask_path, fwhm)

def get_searchlight_RDMs_crossvalidated(data_2d, centers, neighbors, 
                                        obs_descriptor,
                                        method="crossnobis", 
                                        n_conds=None,
                                        method_cov="shrinkage_eye"):

    data_2d, centers = np.array(data_2d), np.array(centers)
    n_centers = centers.shape[0] # number of searchlight centers

    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2)) # create empty RDM array for each point in the searchlight
    for c in range(n_centers):
        # grab this center and neighbors
        center = centers[c]
        nb = neighbors[c]
        # create a database object with this data
        ds = Dataset(data_2d[:, nb],
                     descriptors={"center": c},
                     obs_descriptors=obs_descriptor,
                     channel_descriptors={"voxels": nb}
                     )

        noise_prec_shrink = prec_from_measurements(
            ds, 
            obs_desc="condition", 
            method=method_cov
            )

        RDM_corr = calc_rdm(ds, 
                            method=method,
                            descriptor="condition", 
                            cv_descriptor="run",
                            noise=noise_prec_shrink
                            ).dissimilarities

        RDM[c] = RDM_corr
    return RDMs(
        np.array(RDM),
        rdm_descriptors={"voxel_index": centers},
        dissimilarity_measure=method
        )

print("shape of neural data ...\t", data_2d.shape)
# print("shape of residual data ...\t", data_res.shape)

sl_rdms = get_searchlight_RDMs_crossvalidated(
    data_2d, centers, neighbors,
    obs_descriptor=obs_descriptors,
    method=distance,
    n_conds=len(np.unique(conditions)),
    # residuals=data_res
)

sl_rdms.save(output_folder + filename, file_type="hdf5", overwrite=True)
