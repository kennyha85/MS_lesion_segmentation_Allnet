import os
import sys
import json
import math
import shutil
import numpy as np
import pandas as pd

from os.path import join as pjoin
from multiprocessing import Pool


def registrate(src_path, ref_path, tgt_path, p_id, date_, name):

    tgt_path = pjoin(tgt_path, p_id+'_'+date_+'_'+name)
    if os.path.exists(tgt_path):
        print("p_id %s exits" % p_id)
        return  
    cmd = "/usr/local/fsl/bin/flirt -in " + src_path + " -ref " + ref_path + " -out " + tgt_path + " -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear"

    os.system(cmd)

def coregistration(src_path, tgt_path, p_id):

    src_path = pjoin(src_path, p_id)
    dates = os.listdir(src_path)

    for date_ in dates:
        fp = pjoin(src_path, date_)
        files = os.listdir(fp)
        for file_ in files:
            if 'T2FLAIR' in file_:
                FLAIR_file = file_
                FLAIR_fp = pjoin(fp, file_)
            elif 'T2' in file_:
                T2_fp = pjoin(fp, file_)
            elif 'T1' in file_:
                T1_fp = pjoin(fp, file_)
        tgt_fp = pjoin(tgt_path, p_id, date_)
        os.makedirs(tgt_fp, exist_ok=True)
        registrate(T1_fp, FLAIR_fp, tgt_fp, p_id, date_, 'T1_to_T2FLAIR.nii.gz')
        registrate(T2_fp, FLAIR_fp, tgt_fp, p_id, date_, 'T2_to_T2FLAIR.nii.gz')
        shutil.copyfile(FLAIR_fp, pjoin(tgt_fp, FLAIR_file))

    print("---->>>> Case %s coregistration to FLAIR is done" % (p_id))

def removeSkull(src_fp, tgt_fp, p_id, date_, name):

    tgt_fp = pjoin(tgt_fp, p_id+'_'+date_+'_'+name)
    if os.path.exists(tgt_fp):
        return  
    cmd = "/usr/local/fsl/bin/bet " + src_fp + " " + tgt_fp + " -f 0.5 -g 0"

    os.system(cmd)

def skullRemoval(src_path, tgt_path, p_id):

    src_path = pjoin(src_path, p_id)
    tgt_path = pjoin(tgt_path, p_id)
    dates = os.listdir(src_path)

    for date_ in dates:
        fp = pjoin(src_path, date_)
        tgt_fp = pjoin(tgt_path, date_)
        os.makedirs(tgt_fp, exist_ok=True)
        files = os.listdir(fp)
        for file_ in files:
            src_fp = pjoin(fp, file_)
            if 'T2_to_T2FLAIR' in file_:
                removeSkull(src_fp, tgt_fp, p_id, date_, 'T2_to_T2FLAIR_brain.nii.gz')
            elif 'T1_to_T2FLAIR' in file_:
                removeSkull(src_fp, tgt_fp, p_id, date_, 'T1_to_T2FLAIR_brain.nii.gz')
            elif 'T2FLAIR' in file_:
                removeSkull(src_fp, tgt_fp, p_id, date_, 'T2FLAIR_brain.nii.gz')

    print("---->>>> Case %s skull removal is done" % (p_id))

def run(opt):

    src_path = opt['src_path']
    tgt_path = opt['tgt_path_1']

    p_ids = os.listdir(src_path)
    pool = Pool(processes=12)
    for p_id in p_ids:
        pool.apply_async(coregistration, args=(src_path, tgt_path, p_id))
    
    print("--->>>> All processing to the pool")
    pool.close()
    pool.join()
    print("--->>>> All current processing Finished")

    src_path = opt['tgt_path_1']
    tgt_path = opt['tgt_path_2']

    p_ids = os.listdir(src_path)

    pool = Pool(processes=12)
    for p_id in p_ids:
        pool.apply_async(skullRemoval, args=(src_path, tgt_path, p_id))

    print("--->>>> All processing to the pool")
    pool.close()
    pool.join()
    print("--->>>> All current processing Finished")


if __name__ == "__main__":

    opt = {
        'src_path': './working_data',
        'tgt_path_1': './working_data_registered/',
        'tgt_path_2': './working_pre',
        'n_processes': 12,
    }

    run(opt)