import os
import json
import numpy as np 
import nibabel.processing
import nibabel as nibabel 

from multiprocessing import Pool
from scipy.ndimage.measurements import label as getComponents

def getFileNames(src_path, p_id):

    src_path = os.path.join(src_path, p_id)
    if os.path.isfile(src_path):
        return

    date_ = os.listdir(src_path)[0]
    src_path = os.path.join(src_path, date_)

    files = os.listdir(src_path)
    for file_ in files:
        if "T1_to_T2FLAIR_brain" in file_:
            t1_fp = file_
        elif "T2_to_T2FLAIR_brain" in file_:
            t2_fp = file_
        elif "mat" not in file_:
            fl_fp = file_
        
    return t1_fp, t2_fp, fl_fp, date_

if __name__ == "__main__":
    
    src_path = './working_pre/'
    tgt_path = './data_path.json'
   
    data_json = {}
   
    patient_ids = os.listdir(src_path) 
    for i, p_id in enumerate(patient_ids):
        t1_fp, t2_fp, fl_fp, date_ = getFileNames(src_path, p_id)
        obj = {
            "T1": os.path.join(src_path, p_id, date_, t1_fp),
            "T2": os.path.join(src_path, p_id, date_, t2_fp),
            "FLAIR": os.path.join(src_path, p_id, date_, fl_fp),
        }
        data_json.update({p_id:obj})
        
    with open(tgt_path, 'w') as x:
        json.dump(data_json, x,  indent = 4, 
            sort_keys = True, ensure_ascii = False
        )