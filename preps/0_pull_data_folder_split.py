import os
import sys
import json
import math
import shutil
import numpy as np
import pandas as pd

from os.path import join as pjoin
from multiprocessing import Pool

def matchDate(fp, d_id):

    files = os.listdir(fp)
    for file_ in files:
        if d_id in file_:
            return file_
    
    return None

def cpAndSplitFiles(src_path, tgt_path, p_id, d_id):

    src_fp = pjoin(src_path, p_id)
    date_folder = matchDate(src_fp, d_id)

    if date_folder is None:
        print("---->>>> Failed to find the date %s in subject %s" % (d_id, p_id))

    src_fp = pjoin(src_fp, date_folder)
    files = os.listdir(src_fp)

    tgt_fp = pjoin(tgt_path, p_id, date_folder)
    os.makedirs(tgt_fp, exist_ok=True)

    count = 0
    for file_ in files:
        tgt_file_fp = None
        src_file_fp = pjoin(src_fp, file_)
        if file_ == ("%s_%s_T2FLAIR.nii.gz" % (p_id, date_folder)): 
            tgt_file_fp = pjoin(tgt_fp, file_) 
        elif file_ == ("%s_%s_T2.nii.gz" % (p_id, date_folder)):
            tgt_file_fp = pjoin(tgt_fp, file_) 
        elif file_ == ("%s_%s_T1.nii.gz" % (p_id, date_folder)):
            tgt_file_fp = pjoin(tgt_fp, file_) 
        
        if tgt_file_fp is not None:
            shutil.copyfile(src_file_fp, tgt_file_fp)
            count += 1
    
    print("---->>>> Subject %s of Date %s has %d files" % (p_id, d_id, count))

def run(opt):

    src_path = opt['src_path']
    tgt_path = opt['tgt_path']

    for sample in opt['samples']:
        p_id = sample.split('_')[0]
        d_id = sample.split('_')[1]
        try:
            cpAndSplitFiles(src_path, tgt_path, p_id, d_id)
        except:
            print("---->>>> Subject %s of Date %s has failed" % (p_id, d_id))


def getDataDates(df):

    samples = []
    for idx, row in df.iterrows():
        p_id = str(row['MS ID']).zfill(4)
        date = row['MRI date Baseline']
        year = str(date.year)
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        new_date = year + month + day
        sample = p_id + '_' + new_date
        samples.append(sample)

    return samples

if __name__ == "__main__":

    opt = {
        'src_path': '/MSdata',
        'tgt_path': '/home/hz459/working_data/',
        'n_processes': 12,
        'samples': [], 
        'list_fp': './subject_list.xlsx'
    }

    df = pd.read_excel(opt['list_fp'])
    opt['samples'] = getDataDates(df)


    run(opt)