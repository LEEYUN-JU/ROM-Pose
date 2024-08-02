# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torch

# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

coco_keypoint_pairs = [(16, 14),(14, 12),(17, 15),(15, 13),(12, 13),(6, 12),(7, 13),(6, 7),(6, 8),(7, 9),(8, 10),(9, 11),(2, 3),(1, 2),(1, 3),(2, 4),(3, 5),(4, 6),(5, 7)]

boundary_points = [0,5,11,12,6]

# GPU available check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals

#gpu 
def cal_heat_map(input, input_mask, meta, epoch, output_dir):
    
    if epoch < 30 and epoch >= 0:               
        image_ratio = 0.6; mask_ratio = 0.4;
    if epoch < 50 and epoch >= 30:               
        image_ratio = 0.7; mask_ratio = 0.3;
    if epoch < 80 and epoch >= 50:              
        image_ratio = 0.8; mask_ratio = 0.2;

    inputs = image_ratio * input + mask_ratio * input_mask

    inputs = ((inputs - inputs.min()) / (inputs.max() - inputs.min())) * 255
    
    if inputs is None:
        print("meta_image:", meta['image'], meta['id'].item())
        return None
    

    return inputs

# cpu
def visualize_keypoints(image, keypoints):
    
    try:
        body = []

        for point in boundary_points:
            body.append(keypoints[point])
        
        if isinstance(body, list):
            body = np.array(body, dtype=np.int32)
            
        if isinstance(body, np.ndarray):
            body = np.array(body, dtype=np.int32)

        image = cv2.circle(image, (int(keypoints[0][0]), int(keypoints[0][1])), 80, (255, 255, 255), -1)
        
        #image, pts, isClosed, color, thickness
        image = cv2.polylines(image, [body], True, (255, 255, 255), 30) 
        #boundary_polygon = np.array(body)
        image = cv2.fillPoly(image, [body], color=(255, 255, 255)) 
        
        for pair in coco_keypoint_pairs:
            if keypoints[pair[0]][0] > 0.2 and keypoints[pair[0]][1] > 0.2:
                x1, y1 = int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1])
                x2, y2 = int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1])
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 25)  


        image = cv2.GaussianBlur(image, (15, 15), 10)
    
    except Exception as e:

        print(type(body))
        print(f"exception: {e}")        

    return image
