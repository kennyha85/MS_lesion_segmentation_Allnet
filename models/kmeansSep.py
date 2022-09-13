import os
from numpy.core.defchararray import center
import torch
import numpy as np
import nibabel as nib 

from sklearn.cluster import KMeans
from scipy.ndimage.measurements import label as getComponents

import torch.nn as nn 

class NMS(nn.Module):
    
    def __init__(self, region_size=5,vote_threshold=20):

        super(NMS, self).__init__()

        self.maxpool = nn.MaxPool3d(region_size, stride=1, padding=region_size//2, dilation=1)
        self.vote_threshold = vote_threshold
    
    def forward(self, img_ori):

        img_nms = self.maxpool(img_ori)
        
        return (img_ori==img_nms) * (img_nms > self.vote_threshold)
        # return (img_ori==img_nms)
    
class kmeansSep():
    
    def __init__(self, region_size=7,vote_threshold=0.1):
        
        self.do_nms = NMS(region_size,vote_threshold)
    
    def predict(self, img, msk, lbl=None):
        
        print("---->>>> Total %d counts max %.4f min %.4f..." % (img.sum(),img.max(),img.min()))
        out = self.do_nms(img)
        out = out.squeeze_(0).squeeze_(0).detach().numpy()
        print("---->>>> %d seeds before masking out..." % np.sum(out))
        nms = out * msk
        print("---->>>> %d seeds after masking out..." % np.sum(nms))
        lesion_count = 0
        labels = np.zeros(out.shape)
        lbls, n_lesions = getComponents(msk, np.ones([3,3,3]))
        for idx in range(1, n_lesions+1):
            xs,ys,zs = np.where(lbls==idx)
            centers = []
            feas = []
            for x,y,z in zip(xs,ys,zs):
                feas.append([x,y,z])
                if nms[x,y,z] > 0:
                    centers.append([x,y,z])

            if lbl is None:
                print("---->>>> Total %d centers..." % len(centers))
            else:
                num_actual_lables = len(np.unique(lbl[xs,ys,zs]))
                print("---->>>> Total %d centers actual labels %d..." % (len(centers),num_actual_lables))

            if len(centers)<=1:
                lesion_count += 1
                labels[xs,ys,zs] = lesion_count
            else:
                centers = np.asarray(centers)
                feas = np.asarray(feas)
                
                kmeans = KMeans(n_clusters=centers.shape[0],init=centers,n_init=1).fit(feas)
                k_labels = kmeans.labels_
                
                for j in range(len(xs)):
                    x,y,z = xs[j],ys[j],zs[j]
                    lbl_id = k_labels[j] + lesion_count + 1
                    labels[x,y,z] = lbl_id
                
                lesion_count += len(np.unique(k_labels))
        print("---->>>> Total %d lesions..." % lesion_count)

        return labels