import os
import sys
import argparse
from natsort import natsorted

import numpy as np
import nibabel as nib

""" Set up and interpret command line arguments """
parser = argparse.ArgumentParser(description="Subject-level modeling of fmriprep-preprocessed data")

parser.add_argument("--sub", 
                    help="participant id", type=str)
parser.add_argument("--run", 
                    help="run id", type=str)
parser.add_argument("--task", 
                    help="task id", type=str)
parser.add_argument("--space", 
                    help="space label", type=str)
parser.add_argument("--fwhm", 
                    help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--model_type", 
                    help="trial model scheme (options: `LSA` or `LSS`)", 
                    type=str)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(" ")
    sys.exit(1)
    
subject_id = args.sub
run_id = args.run
task_label = args.task
space_label=args.space
fwhm = args.fwhm
model_type=args.model_type
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

def prep_models_and_args(subject_id=None, run_id=None, task_label=None, fwhm=None, bidsroot=None, 
                         deriv_dir=None, space_label="T1w"):
    from nilearn.glm.first_level import first_level_from_bids

    models, models_run_imgs, \
            models_events, \
            models_confounds = first_level_from_bids(bidsroot, 
                                                     task_label, 
                                                     space_label,
                                                     [subject_id],
                                                     hrf_model="spm",
                                                     standardize="psc",
                                                     t_r=None,
                                                     noise_model="ar1",
                                                     drift_model="cosine",
                                                     drift_order=1,
                                                     img_filters=[("run", run_id), ("desc", "preproc")],
                                                     mask_img=f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_mni/sub-{subject_id}/ses-movie/anat/sub-{subject_id}_ses-movie_label-GM_probseg_binary_mask.nii.gz", 
                                                     smoothing_fwhm=fwhm,
                                                     high_pass=0.008,
                                                     slice_time_ref=None,
                                                     derivatives_folder=deriv_dir,
                                                     minimize_memory=True)

    # fill n/a with 0
    [[mc.fillna(0, inplace=True) for mc in sublist] for sublist in models_confounds]

    # define which confounds to keep as nuisance regressors
    conf_keep_list = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", 
    "a_comp_cor_03", "a_comp_cor_04"
    ]

    """ create events """    
    for sx, sub_events in enumerate(models_events):
        print("sub:\t", models[sx].subject_label)
        for mx, run_events in enumerate(sub_events):
            stim_list = natsorted([str(s) for s in run_events["trial_type"].unique() if str(s) not in ["nan", "None"]])
            run_events["trial_type"] = run_events.trial_type.str.split("_", expand=True)[0]
    return stim_list, models, models_run_imgs, models_events, models_confounds, conf_keep_list

# transform full event design matrix (LSA) into single-event only (LSS)
def lss_transformer(event_df, event_name):
    other_idx = np.array(event_df.loc[:,"trial_type"] != event_name)
    lss_event_df = event_df.copy()
    lss_event_df.loc[other_idx, "trial_type"] = "other_events" 
    return lss_event_df

# ### Run-by-run GLM fit
def nilearn_glm_per_run(stim_list,
                        model_type,
                        models, 
                        models_run_imgs,
                        models_events, 
                        models_confounds,
                        conf_keep_list):
    from nilearn.reporting import make_glm_report
    
    # for each model (corresponding to a subject)
    for midx in range(len(models)):
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim
            print("running GLM with stimulus:\t", stim)

            model = models[midx]

            # set limited confounds
            print("selecting confounds")
            confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]

            # for each run
            for rx in range(len(confounds_ltd)):
                img = models_run_imgs[midx][rx]
                confound = confounds_ltd[rx]
                
                if model_type == "lsa":
                    event = models_events[midx][rx]
                elif model_type == "lss":
                    event = lss_transformer(models_events[midx][rx], stim)
                
                print("trial_target:", sorted(event.trial_type.unique()))
                
                save_path = os.path.join("/home", "data", "study_gaze_tracks", "scratch", f"lss_face-beta", f"sub-{subject_id}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                sub_fname = f"sub-{subject_id}_run-{run_id}_contrast-{contrast_desc}"
                statmap_fpath = os.path.join(save_path, sub_fname + "_beta-map.nii.gz")
                if os.path.isfile(statmap_fpath):
                    print(f"file already exists {statmap_fpath}, abort mission ...")
                    continue
                
                # fit the GLM
                print("fitting GLM on:", img)
                model.fit(img, event, confound)

                # compute the contrast of interest
                statmap = model.compute_contrast(contrast_label, output_type="effect_size")
                nib.save(statmap, statmap_fpath)
                print("saved beta-map to:", statmap_fpath)
                
                # # save report
                # print("saving report")
                # report_fpath = os.path.join(save_path, sub_fname + "_report.html")
                # report = make_glm_report(model=model, 
                #                         contrasts=contrast_label)
                # report.save_as_html(report_fpath)
                # print("saved report to:", report_fpath)
    
""" Multivariate analysis: across-run GLM """
stim_list, models, models_run_imgs, \
    models_events, models_confounds, \
    conf_keep_list = prep_models_and_args(subject_id, 
                                          run_id, 
                                          task_label, 
                                          fwhm, 
                                          bidsroot, 
                                          fmriprep_dir, 
                                          space_label=space_label)

nilearn_glm_per_run(stim_list, 
                    model_type,
                    models,
                    models_run_imgs, 
                    models_events, 
                    models_confounds, 
                    conf_keep_list)