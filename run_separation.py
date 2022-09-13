import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.serialization import load
import nibabel as nib

from models.utils import getDevice, loadSepModel, parseJsonAndGetDataPath, loadAndPrepData, reshapeToOriSizeWithCrop, loadPredMask
from models.kmeansSep import kmeansSep

def run_segmentation(opt): 

    os.makedirs(opt["save_path"],exist_ok=True)
    device = getDevice(opt)
    model = loadSepModel(device)
    model.eval()
    
    kmeans_sep = kmeansSep()
    
    st = time.time()
    file_fp = {
        'T1': opt['t1_fp'],
        'T2': opt['t2_fp'],
        'FLAIR': opt['fl_fp'],
    }

    img, header, affine, crop_pos = loadAndPrepData(file_fp)
    mask = loadPredMask(0, opt['msk_fp'], crop_pos)
    mask = (mask>0).astype(float)
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    img_size = mask.shape
    
    with torch.no_grad():
        out = model(img)
        out = out[1]    
    out = [x.squeeze_(0).squeeze_(0).detach().cpu().numpy() for x in out]
    hvote = torch.zeros(img_size)
    
    for ii in range(img_size[0]):
        for jj in range(img_size[1]):
            for kk in range(img_size[2]):
                if mask[ii,jj,kk] > 0:
                    x,y,z,w = out[0][ii,jj,kk],out[1][ii,jj,kk],out[2][ii,jj,kk],out[3][ii,jj,kk]
                    x,y,z = int(np.round(x)),int(np.round(y)),int(np.round(z))
                    x = min(max(0,ii-x),img_size[0]-1)
                    y = min(max(0,jj-y),img_size[1]-1)
                    z = min(max(0,kk-z),img_size[2]-1)
                    hvote[x,y,z] += w
                    
    hvote.unsqueeze_(0).unsqueeze_(0)
    pred = kmeans_sep.predict(hvote,mask)
    pred = reshapeToOriSizeWithCrop(pred, crop_pos)
    
    if '.nii.gz' in opt['save_path'] or '.nii' in opt['save_path']:
        save_fp = opt['save_path']
    else:
        save_fp = opt['fl_fp'].split('/')[-1].split('.nii')[0]+'_label.nii.gz'
        save_fp = os.path.join(opt['save_path'], save_fp)
        
    nib.save(nib.Nifti1Image(pred, affine, header), save_fp)    
    print("---->>>> Lesion separation %s is processed, using %.2fs" % (save_fp, time.time()-st))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "ms lesion segmentation")
    parser.add_argument("--t1_fp", type = str)
    parser.add_argument("--t2_fp", type = str)
    parser.add_argument("--fl_fp", type = str)
    parser.add_argument("--msk_fp", type = str)
    parser.add_argument("--save_path", type = str, default = "./segmentations")
    parser.add_argument("--gpu_id", type = int, default = -1)  # default no gpu
    
    opt = {**vars(parser.parse_args())}
    
    run_segmentation(opt)
    
    #python run_separation_cmd.py --gpu_id -1 --t1_fp /home/hz459/segmentation_inference/data/0460/0460_20181209_163020_T1_to_T2FLAIR_brain.nii.gz --t2_fp /home/hz459/segmentation_inference/data/0460/0460_20181209_163020_T2_to_T2FLAIR_brain.nii.gz --fl_fp /home/hz459/segmentation_inference/data/0460/0460_20181209_163020_T2FLAIR_brain.nii.gz --msk_fp /home/hz459/segmentation_inference/segmentations/0460_20181209_163020_T2FLAIR_brain_mask.nii.gz