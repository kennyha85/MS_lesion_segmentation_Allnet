import os
import time
import torch
import argparse
import nibabel as nib

from models.utils import getDevice, loadSegModel, loadAndPrepData, reshapeToOriSizeWithCrop


def run_segmentation(opt): 

    os.makedirs(opt["save_path"],exist_ok=True)
    device = getDevice(opt)
    model = loadSegModel(device)
    model.eval()
    
    print("--------------------runing segmentation--------------------")
    file_fp = {
        'T1': opt['t1_fp'],
        'T2': opt['t2_fp'],
        'FLAIR': opt['fl_fp'],
    }
    st = time.time()
    img, header, affine, crop_pos = loadAndPrepData(file_fp)
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        msk = model(img)
    msk = (msk.squeeze(0).squeeze(0).sigmoid() >= opt["threshold"])
    msk = msk.detach().cpu().numpy().astype(float)
    msk = reshapeToOriSizeWithCrop(msk,crop_pos)
    
    if '.nii.gz' in opt['save_path'] or '.nii' in opt['save_path']:
        save_fp = opt['save_path']
    else:
        save_fp = opt['fl_fp'].split('/')[-1].split('.nii')[0]+'_mask.nii.gz'
        save_fp = os.path.join(opt['save_path'], save_fp)
        
    nib.save(nib.Nifti1Image(msk, affine, header), save_fp)    
    print("---->>>> Segmentation %s is processed, using %.2fs" % (save_fp, time.time()-st))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "ms lesion segmentation")
    parser.add_argument("--t1_fp", type = str)
    parser.add_argument("--t2_fp", type = str)
    parser.add_argument("--fl_fp", type = str)
    parser.add_argument("--save_path", type = str, default = "./segmentations")
    parser.add_argument("--threshold", type = float, default = 0.01)
    parser.add_argument("--gpu_id", type = int, default = -1)  # default no gpu
    
    opt = {**vars(parser.parse_args())}
    
    run_segmentation(opt)
    
    #python run_segmentation.py --threshold 0.5 --gpu_id -1 --t1_fp /home/hz459/segmentation_inference/data/0460/0460_20181209_163020_T1_to_T2FLAIR_brain.nii.gz --t2_fp /home/hz459/segmentation_inference/data/0460/0460_20181209_163020_T2_to_T2FLAIR_brain.nii.gz --fl_fp /home/hz459/segmentation_inference/data/0460/0460_20181209_163020_T2FLAIR_brain.nii.gz
