import re
import glob 
import itertools
from nilearn.image import concat_imgs

# housekeeping
sub_list = ["%.2d" % i for i in range(1, 21)]
ha_runs = ["2", "4", "7", "8"]

# delete missing subjects from combi dataset
sub_list.remove("07")
sub_list.remove("08")
sub_list.remove("11")
sub_list.remove("12")
sub_list.remove("13")

betas_dir = "/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/nibetaseries"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r"(\d+)", text) ]

for sub in sub_list:
    a_flist = []

    for run in ha_runs:
        run_flist = glob.glob(betas_dir + f"/sub-{sub}/ses-movie/func/sub-{sub}_ses-movie_task-movie_run-{run}_space-MNI152NLin2009cAsym_desc-chunk*_betaseries.nii.gz")
        run_flist.sort(key=natural_keys)
        a_flist.append(run_flist)
    
    conc_list = list(itertools.chain.from_iterable(a_flist))
    print(f"number of elements in list for sub-{sub} ... \t", len(conc_list))
    concat_imgs(conc_list).to_filename(betas_dir + f"/sub-{sub}_ha-input.nii.gz")