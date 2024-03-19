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

# COCO 키포인트 인덱스와 쌍을 정의합니다.
coco_keypoint_pairs = [(16, 14),(14, 12),(17, 15),(15, 13),(12, 13),(6, 12),(7, 13),(6, 7),(6, 8),(7, 9),(8, 10),(9, 11),(2, 3),(1, 2),(1, 3),(2, 4),(3, 5),(4, 6),(5, 7)]
# 경계선 좌표
boundary_points = [0,5,11,12,6]

# GPU를 사용할 수 있는지 확인
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

#gpu 연산
def cal_heat_map(input, input_mask, meta, epoch, output_dir):
    
    if epoch < 30 and epoch >= 0:               
        image_ratio = 0.6; mask_ratio = 0.4;
    if epoch < 50 and epoch >= 30:               
        image_ratio = 0.7; mask_ratio = 0.3;
    if epoch < 80 and epoch >= 50:              
        image_ratio = 0.8; mask_ratio = 0.2;
    # if epoch < 100 and epoch >= 80:             
    #     image_ratio = 0.9; mask_ratio = 0.1;
    # if epoch >= 100:             
    #     return None;
   
    #print("image, heatmap", image_ratio, heatmap_ratio)
    
    #inputs = cv2.addWeighted(input, image_ratio, input_mask, mask_ratio, 0, dtype=cv2.CV_64F)            
    # 텐서 간의 가중합 수행 (element-wise multiplication 후 sum)
    inputs = image_ratio * input + mask_ratio * input_mask
    
    # 결과를 [0, 255] 범위로 조정 (정규화)
    inputs = ((inputs - inputs.min()) / (inputs.max() - inputs.min())) * 255
    
    if inputs is None:
        print("meta_image:", meta['image'], meta['id'].item())
        return None
    
    # if epoch % 10 == 0:
    #     cv2.imwrite(output_dir + '/input_img.jpg', inputs.cpu().numpy())
    #     print("image, heatmap", image_ratio, mask_ratio)
    
    return inputs

# cpu연산
def visualize_keypoints(image, keypoints):
    
    #키포인트가 0으로 추정되는 경우 마스크를 씌우지 않고 그냥 원본 이미지를 반환하게 됌
    try:
        body = []

        #몸통 좌표 구하기
        for point in boundary_points:
            body.append(keypoints[point])
        
        if isinstance(body, list):
            body = np.array(body, dtype=np.int32)
            
        if isinstance(body, np.ndarray):
            body = np.array(body, dtype=np.int32)
        
        #image, center, radius, color, thickness
        #thickness가 -1이면 원이 채워지게 된다.
        image = cv2.circle(image, (int(keypoints[0][0]), int(keypoints[0][1])), 80, (255, 255, 255), -1)
        
        #image, pts, isClosed, color, thickness
        image = cv2.polylines(image, [body], True, (255, 255, 255), 30) #각 keypoints를 잇는 선 그려주기

        #boundary_polygon = np.array(body)
        image = cv2.fillPoly(image, [body], color=(255, 255, 255)) #선 색을 하얀색으로 지정해 주는 이유, 색이 겹치면 겹치는 부분이 지워지기 때문

        # 키포인트 쌍을 이용하여 연결선 그리기
        for pair in coco_keypoint_pairs:
            if keypoints[pair[0]][0] > 0.2 and keypoints[pair[0]][1] > 0.2:
                x1, y1 = int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1])
                x2, y2 = int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1])
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 25)  # 하얀색으로 그리기

        # 선 주변에 가우시안 블러 적용
        image = cv2.GaussianBlur(image, (15, 15), 10)
    
    except Exception as e:
        # 예외 처리 코드
        print(type(body))
        print(f"예외가 발생했습니다: {e}")        

    return image


# #배치 사이즈 만큼의 개수가 들어감
# def cal_heat_map(meta, mask_info, output_dir, epoch):
#     batch_size = len(meta['image'])
#     transform = transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ])
    
#     inputs = []  # GPU로 이동할 이미지 배치를 저장할 리스트
    
#     #print("cal_heat_map len_meta_image", len(meta['image'])) #192

#     for i in range(0, batch_size):
        
#         if meta['image'][i] in mask_info:            
#             pred = mask_info[meta['image'][i]]
#             image_file = meta['image'][i]            

#             # 원본 이미지 불러오기
#             data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

#             # 원본 이미지 사이즈 재기
#             height, width, channels = data_numpy.shape        

#             # 이미지와 동일한 크기의 빈 3차원 배열 생성
#             sum_heatmap = np.zeros((height, width, channels), dtype=np.uint8)
            
#             # print("preds size", pred.size) #6528
#             # print("preds_len", len(pred)) #192
#             # print("preds_i", pred[0]) #[[192], [192]]
#             # print("preds_i_len", len(pred[0])) #2
#             # print("preds_i_len_0", len(pred[0][0])) #17
            
#             for j in range(0, len(pred)):                
#                 # 개별 관절 위치에 원 그리기
#                 # cv2.circle(배경이미지, (x좌표값, y좌표값), 원주, 색상, 원 채움 여부(-1: 꽉 채우기, 1:안채우기))
#                 cv2.circle(sum_heatmap, (int(pred[j][0]), int(pred[j][1])), 40, [255, 255, 255], -1) #원본 이미지를 나타내는 부분

#             # 키포인트 선으로 연결하기
#             sum_heatmap = visualize_keypoints(sum_heatmap, pred)
            
# #             data_numpy = visualize_keypoints(data_numpy, pred)            
# #             # sum_heatmap 이미지를 저장
# #             cv2.imwrite(output_dir + f'/sum_heatmap_{i}.jpg', sum_heatmap)            
# #             # sum_heatmap 이미지를 저장
# #             cv2.imwrite(output_dir + f'/data_numpy_{i}.jpg', data_numpy)
            
#             # mask 이미지를 binary 이미지로 만들어주기
# #             sum_heatmap = np.where(sum_heatmap > 0, 0, 0)
            
# #             sum_heatmap = sum_heatmap.astype(np.float64)
# #             data_numpy = data_numpy.astype(np.float64)
    
#             # 그레이 스케일로 변환
#             gray_heatmap = cv2.cvtColor(sum_heatmap, cv2.COLOR_BGR2GRAY)
            
#             # 임계값 이진화 수행 (예: 임계값 50 사용)
#             _, gray_heatmap = cv2.threshold(gray_heatmap, 50, 255, cv2.THRESH_BINARY)
            
#             #다시 3차원 채널로 변경해줌 > 3차원 채널이 아니면 출력이 제대로 되지 않음
#             gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_GRAY2BGR) 
            
#             # 원본 이미지와 마스크 이미지 합치기
#             if epoch > 50:               
#                 sum_heatmap = cv2.addWeighted(data_numpy, 0.2, sum_heatmap, 0.8, 0, dtype=cv2.CV_64F)
#             else:
#                 sum_heatmap = cv2.addWeighted(data_numpy, 0.4, sum_heatmap, 0.6, 0, dtype=cv2.CV_64F)
            
#             # if i % 10 == 0:
#             #     cv2.imwrite(output_dir + f'/data_numpy_{i}.jpg', sum_heatmap)

#             #원래 모델 입력 사이즈 및 affine을 다시 수행해 주어야 함
#             c = meta['center'][i]
#             s = meta['scale'][i]
#             r = meta['rotation'][i]            

#             # 데이터를 부동 소수점으로 변환
#             sum_heatmap = sum_heatmap.astype(np.float64)
            
#             # print("c", c)
#             # print("S", s)
            
#             trans = get_affine_transform(c, s, r, (192, 256))
            
#             # 데이터를 부동 소수점으로 변환
#             trans = trans.astype(np.float64)
            
# #             print(sum_heatmap)
# #             print(trans)
            
# #             raise ValueError("hihihi")
                
#             input = cv2.warpAffine(
#                 sum_heatmap,
#                 trans,
#                 (192, 256),
#                 flags=cv2.INTER_LINEAR)  

#             input = transform(input)
            
#             input = (input - input.min()) / (input.max() - input.min())
        
#             inputs.append(input)
            
#         else:
#             print("meta_image:", meta['image'][i])
            
#             #print("mask_info:", mask_info)
#             # print("mask_type:", type(str(meta['image'][i])))            
#             return None
            
    
#     # GPU로 이동할 이미지 배치 생성
#     inputs = torch.stack(inputs)
    
#     # print("Type of input:", type(inputs))   #torch.tensor
#     # print("Shape of input:", inputs.shape)  #192, 3, 256, 192
    
#     # Step 1: Torch 텐서를 NumPy 배열로 변환
#     # Change the tensor's shape from [192, 3, 256, 192] to [192, 256, 192, 3] and move it to CPU if needed
#     input_numpy = inputs.permute(0, 2, 3, 1).cpu().numpy()  

#     # Step 2: NumPy 배열을 OpenCV 이미지로 변환
#     #input_opencv = (input_numpy[0] * 255).astype(np.uint8)  # Convert the first image in the batch to 8-bit unsigned integer (0-255)
    
#     # Step 2: NumPy 배열 중에서 하나의 이미지로 결합
#     combined_image = np.concatenate(input_numpy, axis=0)  # 배치의 이미지들을 하나로 결합

#     # Step 3: OpenCV 이미지로 변환
#     combined_opencv = (combined_image * 255).astype(np.uint8)  # 8비트 부호 없는 정수 (0-255)로 변환
    
#     # 이미지를 OpenCV를 사용하여 저장
#     if epoch % 10 == 0:
#         cv2.imwrite(output_dir + '/input_img.jpg', combined_opencv)  # 이미지를 OpenCV로 저장
#     #print("img saved")    

#     return inputs

