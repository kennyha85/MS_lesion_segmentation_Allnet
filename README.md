# Lesion Segmentation

run following command as an example

'''
python run_segmentation.py --threshold 0.5 --gpu_id -1 --t1_fp ./data/0460/0460_20181209_163020_T1_to_T2FLAIR_brain.nii.gz --t2_fp ./data/0460/0460_20181209_163020_T2_to_T2FLAIR_brain.nii.gz --fl_fp ./data/0460/0460_20181209_163020_T2FLAIR_brain.nii.gz
'''

--threshold: a number between 0 and 1
--t1_fp: path for T1w image
--t2_fp: path for T2w image
--fl_fp: path for FLAIR image
--gpu_id: -1 stands for CPU onlu, 0 and other positive integer denotes for GPU id


# Lesion Separartion

run following command as an example


'''
python run_separation.py --gpu_id -1 --t1_fp ./data/0460/0460_20181209_163020_T1_to_T2FLAIR_brain.nii.gz --t2_fp ./data/0460/0460_20181209_163020_T2_to_T2FLAIR_brain.nii.gz --fl_fp ./data/0460/0460_20181209_163020_T2FLAIR_brain.nii.gz --msk_fp ./segmentations/0460_20181209_163020_T2FLAIR_brain_mask.nii.gz
'''

please note that the segmentation mask is a required for lesion separation

--t1_fp: path for T1w image
--t2_fp: path for T2w image
--fl_fp: path for FLAIR image
--msk_fp: path for segmentation mask
--gpu_id: -1 stands for CPU onlu, 0 and other positive integer denotes for GPU id
