import glob
import datalad.api as dl

vis_rois_path = "/home/data/study_gaze_tracks/studyforrest-data-visualrois"
dl.clone("https://github.com/psychoinformatics-de/studyforrest-data-visualrois", path=vis_rois_path)
ds = dl.Dataset(vis_rois_path)
rois_path = glob.glob(vis_rois_path + "/sub-*/rois/*.nii.gz")
result = dl.get(dataset=rois_path, recursive=True)