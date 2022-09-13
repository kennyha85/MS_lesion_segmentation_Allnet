The source code for MS lesion segmentation with Allnet

Reference: "Zhang, H., Zhang, J., Li, C., Sweeney, E.M., Spincemaille, P., Nguyen, T.D., Gauthier, S.A., Wang, Y. and Marcille, M., 2021. ALL-Net: Anatomical information lesion-wise loss function integrated into neural network for multiple sclerosis lesion segmentation. NeuroImage: Clinical, 32, p.102854."
link: https://www.sciencedirect.com/science/article/pii/S2213158221002989

Inputs: the network requires T1w,T2w and FLAIR images as the inputs. Note that T1w and T2w images should be registed to the FLAIR image space.

You can download the trained model for MS lesion segmentation here: https://drive.google.com/file/d/1gG3bn9nbLg0pBQ4FIJ8eo0erUEM1_pi7/view?usp=sharing

The trained model for MS lesion separation can be downloaded here: https://drive.google.com/file/d/1fCtkN-X60NjK81DgTm-1PlKfWQM5O7Uq/view?usp=sharing

The images should be put in "data/pid" and the trained models should be put in "models" folder. 



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
