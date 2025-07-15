import os
import sys
import glob
import numpy as np
import pandas as pd
from natsort import natsorted
from nilearn.image import load_img, binarize_img, resample_to_img
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.compare import compare
from rsatoolbox.rdm.calc import calc_rdm
import warnings

warnings.simplefilter("ignore", UserWarning)

# I/O
run = sys.argv[1]
distance_metric = "sqeuclidean"
sub_list = ["01", "02", "03", "04", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]
chunk = [0, 59, 68, 55, 71, 54, 68, 83, 51][int(run)]
hypothesis_list = [
    "spatial_distribution", 
    "temporal", 
    "fixation_dur",
    "fix_cnt",
    "sacc_amp"] 
print(f"loading data for run-{run}, scenes available {chunk}")

# paths
deriv_fpath = os.path.join(os.getcwd(), "derivatives")
fixation_dur_path = os.path.join(deriv_fpath, "model_rdms", "chunk_wise", "fix_dur")
fixation_cnt_path = os.path.join(deriv_fpath, "model_rdms", "chunk_wise", "fix_cnt")
sacc_amp_path = os.path.join(deriv_fpath, "model_rdms", "chunk_wise", "sacc_amp")
temporal_path = os.path.join(deriv_fpath, "model_rdms", "chunk_wise", "temporal", f"{distance_metric}")
spatial_path = os.path.join(deriv_fpath, "model_rdms", "chunk_wise", "spatial_distribution", f"{distance_metric}")
example_img = os.path.join(deriv_fpath, "lss_spatial-attention_beta", "sub-01", "sub-01_run-1_contrast-chunk1_beta-map.nii.gz")

# ROI masks
roi_mask_list = glob.glob(os.path.join(deriv_fpath, "roi_masks", "chunk_wise", "*.nii.gz"))
roi_names = [os.path.basename(roi).split(".")[0] for roi in roi_mask_list]

# output directory
out_dir_roi = os.path.join(deriv_fpath, "chunk-wise_roi-results")
os.makedirs(out_dir_roi, exist_ok=True)
output_tsv_path = os.path.join(out_dir_roi, f"run-{run}_results.tsv")

def upper_tri(RDM):
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

# initial.
all_results = []

for hypothesis_rdm in hypothesis_list:
    results_dict = {"hypothesis": [], "scene": []}
    
    for roi_name in roi_names:
        results_dict[roi_name] = []

    for scene in range(1, chunk + 1):
        scene_str = f"{scene:02d}"
        results_dict["hypothesis"].append(hypothesis_rdm)
        results_dict["scene"].append(scene)

        for roi_fpath, roi_name in zip(roi_mask_list, roi_names):
            print(f"Processing scene-{scene_str}, ROI: {roi_name}")

            # load and resample mask
            tmp_roi = resample_to_img(
                binarize_img(roi_fpath, copy_header=True), 
                example_img, 
                copy_header=True, 
                interpolation="nearest").get_fdata()
            
            target_mask = (tmp_roi == 1)
            roi_size = target_mask.sum()

            # load beta maps
            nifti_fpaths = natsorted(glob.glob(os.path.join(
                deriv_fpath, "lss_spatial-attention_beta", "sub-*", f"sub-*run-{run}_contrast-chunk{scene}_beta-map.nii.gz"
            )))
            patterns = np.full((len(nifti_fpaths), roi_size), np.nan)
            
            for c, nifti_img in enumerate(nifti_fpaths):
                patterns[c, :] = load_img(nifti_img).get_fdata()[target_mask].squeeze()

            # create dataset
            ds = Dataset(
                measurements=patterns,
                descriptors={"run": run, "task": "study_face_tracks"},
                obs_descriptors={"condition": sub_list}
            )

            # compute neural RDM
            neural_rdm = calc_rdm(ds, method="euclidean", descriptor="condition")

            # load specific hypothesis rdm
            if hypothesis_rdm == "fixation_dur":
                model_rdm_path = os.path.join(fixation_dur_path, f"ses-movie_task-movie_run-0{run}_chunk{scene}_rdm-fixdur_dist-diff.npy")
            elif hypothesis_rdm == "fixation_cnt":
                model_rdm_path = os.path.join(fixation_cnt_path, f"ses-movie_task-movie_run-0{run}_chunk{scene}_rdm-fixcnt_dist-diff.npy")
            elif hypothesis_rdm == "sacc_amp":
                model_rdm_path = os.path.join(sacc_amp_path, f"ses-movie_task-movie_run-0{run}_chunk{scene}_rdm-saccamp_dist-diff.npy")
            elif hypothesis_rdm == "temporal":
                model_rdm_path = os.path.join(temporal_path, f"ses-movie_task-movie_run-0{run}_chunk{scene}_rdm-{distance_metric}.npy")
            elif hypothesis_rdm == "spatial_distribution":
                model_rdm_path = os.path.join(spatial_path, f"ses-movie_task-movie_run-0{run}_chunk{scene}_rdm-{distance_metric}.npy")

            model_rdm = upper_tri(np.load(model_rdm_path))

            # compute correlation
            raw_corr = compare(model_rdm, neural_rdm, method="spearman")[0][0]

            # apply Fisher Z-transformation
            result = np.arctanh(np.clip(raw_corr, -0.999999, 0.999999))
            results_dict[roi_name].append(result)

    # append current hypothesis
    all_results.append(pd.DataFrame(results_dict))

# concatenate results
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(output_tsv_path, sep="\t", index=False)
print(f"Results saved to {output_tsv_path}")
