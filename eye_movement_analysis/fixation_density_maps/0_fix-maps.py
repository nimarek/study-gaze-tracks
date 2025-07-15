import os
from eyegaze_utils import movie_dataset, preprocess_eyegaze

# sub_list = [5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]
sub_list = [31, 32, 33, 34, 35, 36] # 22, 23, 24, 25, 26, 27, 28, 29, 30, 

for sub in sub_list:
    print("working on sub %s" % (sub))
    base_path="/home/data/study_gaze_tracks/derivatives/fix_maps"
    fname_tmpl="/sub-%s_ses-movie_task-movie" % (sub)

    df = movie_dataset(sub)
    df.to_npz(base_path + fname_tmpl)