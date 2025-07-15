#!/usr/bin/env python

import os
import re
import sys
import time
import glob

import numpy as np
from mvpa2.datasets.base import Dataset
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
from mvpa2.mappers.zscore import zscore
from mvpa2.datasets.mri import fmri_dataset

prep_bool = False

# execute only once to generate hdf5 data
sub = str(sys.argv[1])


# set paths
prep_dir = "/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep"
nibeta_dir = "/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/nibetaseries"
output_parameters_dir = "/home/data/study_gaze_tracks/derivatives/hyperalignment"

if not os.path.exists(output_parameters_dir):
    os.makedirs(output_parameters_dir)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def prepare_subject(sub, out_dir, is_beta=False):
    """
    Function to convert niftis to hdf5.
    """
    if is_beta == True:
        in_dir = "/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/nibetaseries"
        out_fname = out_dir + "/beta_lss/data_sets/sub-%s_ses-movie_task-movie_space-MNI152NLin2009cAsym.hdf5" % sub
        bold_fname = in_dir + "/sub-%s_ha-input.nii.gz" % sub

        if not os.path.exists(out_dir + "/beta_lss/transf-matrices"):
            os.makedirs(out_dir + "/beta_lss/transf-matrices")
            out_dir = os.path.join(out_dir, "/beta_lss/transf-matrices")
    else:
        in_dir = "/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep"
        out_fname = out_dir + "/epi_isc/data_sets/sub-%s_ses-movie_task-movie_run-%s_space-MNI152NLin2009cAsym.hdf5" % (sub, run)
        bold_fnames = glob.glob(in_dir + "/sub-%s/ses-movie/func/sub-%s_ses-movie_task-movie_run-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % (sub, sub, run))
        
        if not os.path.exists(out_dir + "/epi_isc/transf-matrices"):
            os.makedirs(out_dir + "/epi_isc/transf-matrices")
            out_dir + "/epi_isc/transf-matrices"

    mask_fname = "/home/data/study_gaze_tracks/code/mni_mask.nii"
    ds = fmri_dataset(samples=bold_fname, mask=mask_fname, chunks=np.arange(1, 234 + 1))
    zscore(ds, chunks_attr=None)

    print("saving to %s ..." % out_fname)
    h5save(out_fname, ds)

    return None

def run_hyperalignment( sl_radius, out_dir, is_beta=False):
    """
    Execute pymvpas searchlight hyperalignment procedure. Compute reverse mapper for each subject
    from common-space to subject space.
    """
    ds_all = []
    sub_list = ["%.2d" % i for i in range(1, 21)]
    
    # delete missing subjects from combi dataset
    sub_list.remove("07")
    sub_list.remove("08")
    sub_list.remove("11")
    sub_list.remove("12")
    sub_list.remove("13")
        
    for sub in sub_list:
        if is_beta == True:
            print "%s/beta_lss/data_sets/sub-%s_ses-movie_task-movie_space-MNI152NLin2009cAsym.hdf5" % (out_dir, sub)
            ds_all.append(h5load("%s/beta_lss/data_sets/sub-%s_ses-movie_task-movie_space-MNI152NLin2009cAsym.hdf5" % (out_dir, sub)))
            hmappers_fname = output_parameters_dir + "/beta_lss/transf-matrices/combined_sl-radius-%s_hmappers.hdf5" % sl_radius
        else:
            print "working on: %s/epi_isc/beta_lss/sub-%s_ses-movie_task-movie_run-%s_space-MNI152NLin2009cAsym.hdf5" % (out_dir, sub, run)
            ds_all.append(h5load("%s/epi_isc/beta_lss/sub-%s_ses-movie_task-movie_run-%s_space-MNI152NLin2009cAsym.hdf5" % (out_dir, sub, in_runs)))
            hmappers_fname = output_parameters_dir + "/epi_isc/transf-matrices/combined_sl-radius-%s_hmappers.hdf5" % sl_radius
    
    sl_hyperal = SearchlightHyperalignment(radius=sl_radius, sparse_radius=5, compute_recon=True, dtype="float16")
    hmappers = sl_hyperal(ds_all)

    print("saving to %s ..." % hmappers_fname)
    h5save(hmappers_fname, hmappers, mode="a")
       
    return None

if prep_bool == True:
    # (1) generate hdf5 files per subject per run
    prepare_subject(sub=sub, out_dir=output_parameters_dir, is_beta=True)
else:
    # (2) load preprocessed data and run hyperalignment. Spit out mappers.
    slhyper_start_time = time.time()
    hmappers = run_hyperalignment(sl_radius=sl_radius, out_dir=output_parameters_dir, is_beta=True)
    print("done in %.1f seconds" % (time.time() - slhyper_start_time))