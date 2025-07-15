import sys
import glob
import os
import itertools

import numpy as np
import nibabel as nib
from scipy.sparse import save_npz, load_npz
from scipy.stats import zscore as zscore_matrix
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.mappers.zscore import zscore
from mvpa2.datasets.mri import fmri_dataset, map2nifti

sub_list = ["%.2d" % i for i in range(1, 21)]
non_ha_runs = ["1", "3", "5", "6"]
sl_radius = str(sys.argv[1])
input_dir = "/home/data/study_gaze_tracks/derivatives/hyperalignment/beta_lss"

# delete missing subjects from combi dataset
sub_list.remove("07")
sub_list.remove("08")
sub_list.remove("11")
sub_list.remove("12")
sub_list.remove("13")
        
def slice_mapper(sub_list, sl_radius, in_dir):
    """
    Load mapper for all, seperate and store rotation matrices as .npz file for each subject.
    """
    aligned_betas_dir = "/home/data/study_gaze_tracks/derivatives/hyperaligned_betas"
    if not os.path.exists(aligned_betas_dir):
        os.makedirs(aligned_betas_dir)

    transf_mappers = h5load("%s/transf-matrices/combined_sl-radius-%s_hmappers.hdf5" % (in_dir, sl_radius))

    for mapper, sub in zip(transf_mappers, sub_list):
        print "saving transformation matrix of sub-%s ..." % sub
        save_npz(in_dir + "/transf-matrices/sub-{s}_sl_radius-{l}_ha_mapper.npz".format(s=sub, l=sl_radius), mapper._proj)
        h5save(in_dir + "/transf-matrices/sub-{s}_sl_radius-{l}_ha_mapper.hdf5".format(s=sub, l=sl_radius), mapper._proj, mode="a")

    return None

def transform_betas(sub_list, run, sl_radius, in_dir):
    """
    Load stored hyperalignment mapper and z-scored beta-files per subject and run. Project 
    data into common space, apply z-scoring again and map back to subject space. Save the 
    results in the derivatives folder.
    """
    nibeta_dir = "/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/nibetaseries"
    out_dir = "/home/data/study_gaze_tracks/derivatives/hyperaligned_betas"

    transf_mappers = h5load(in_dir + "/transf-matrices/combined_sl-radius-%s_hmappers.hdf5" % sl_radius)
    mask_fname = "/home/data/study_gaze_tracks/code/mni_mask.nii"
    chunk_transf = 0

    for sub in sub_list:
        print "working on sub-%s and run-%s ..." % (sub, run)
        
        if not os.path.exists(out_dir + "/sub-%s" % sub):
            os.makedirs(out_dir + "/sub-%s" % sub)

        beta_lss = glob.glob("%s/sub-%s/ses-movie/func/sub-%s_ses-movie_task-movie_run-%s_space-MNI152NLin2009cAsym_desc-chunk*_betaseries.nii.gz" % (nibeta_dir, sub, sub, run))
        transf_mapper = transf_mappers[chunk_transf]
        chunk_beta = 0

        for beta_img in beta_lss:
            ds = fmri_dataset(samples=beta_img, mask=mask_fname)
            ds_hyperaligned = transf_mapper.forward(ds)
            zscore_matrix(ds_hyperaligned, axis=0)

            nifti_tmp = map2nifti(dataset=ds_hyperaligned, imghdr=ds.a.imghdr)
            chunk_beta += 1
            nifti_tmp.to_filename("%s/sub-%s/sub-%s_ses-movie_task-movie_run-%s_sl-radius-%s_chunk%s_space-hyperaligned.nii.gz" % (out_dir, sub, sub, run, sl_radius, chunk_beta))
        
        chunk_transf += 1

    return None

# load hyperalignment mappers. Spit out niftis.
slice_mapper(sub_list=sub_list, sl_radius=sl_radius, in_dir=input_dir)

for run in non_ha_runs:
    transform_betas(sub_list=sub_list, run=run, sl_radius=sl_radius, in_dir=input_dir)