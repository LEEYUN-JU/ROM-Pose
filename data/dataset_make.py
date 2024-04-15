# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:35:28 2022

@author: labpc
"""

import json
import numpy as np
import math
import cv2
from PIL import Image
from pycocotools.coco import COCO
import os
from PIL import Image
from matplotlib import pyplot as plt

def bring_json(base_path, coordinate):  

    with open('../../../../home/yju/tf/data/coco/annotations/person_keypoints_train2017.json' ,  'r') as f:
        coordinate = json.load(f) 
    
    coordinate = coordinate['annotations']
    
    temp = [] 
    for i in range(0, len(coordinate)):
        if coordinate[i]['num_keypoints'] > 0:
            temp.append(coordinate[i])
    
    coordinate = temp

    return coordinate


############################## mask 이미지 그리기 ###############################
def draw_masking():    
    coco = COCO('../../../../home/yju/tf/data/coco/annotations/person_keypoints_train2017.json')
    img_dir = '../../../../home/yju/tf/data/coco/images/train2017'
    
    coco_keys = coco.imgs.keys()    

    for k, j in enumerate(coco_keys): 
        
        if k>0:
            break        
        img = coco.imgs[j]
        
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)            
        
        for i in range(len(anns)):                 
            mask = coco.annToMask(anns[i])
            temp = Image.fromarray(mask*255)  
            temp.save("../../../../home/yju/tf/data/%012d_%04d.jpg"%(anns[i]['image_id'], anns[i]['id']), cmap='gray')           

def draw_skeleton(bbox=None):
    connection = [[16, 14], [14, 12], [17, 15], 
                  [15, 13], [12, 13], [6, 12], 
                  [7, 13], [6, 7], [6, 8], 
                  [7, 9], [8, 10], [9, 11], 
                  [2, 3], [1, 2], [1, 3], 
                  [2, 4], [3, 5], [4, 6], [5, 7]]

    colors = [[255, 255, 255]] * len(connection)

    coco = COCO('../../../../home/yju/tf/data/coco/annotations/person_keypoints_train2017.json')
    coco_keys = list(coco.imgs.keys())

    for k, j in enumerate(coco_keys):    

        if k > 0:
            break

        img = coco.imgs[j]

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        for i in range(len(anns)): 
            kpt = anns[i]['keypoints']
            kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
            npart = kpt.shape[0]

            img = plt.imread("../../../../home/yju/tf/data/coco/images/train2017/%012d.jpg"%anns[i]['image_id'])
            canvas = img.copy()

            if npart == 17: # ochuman, data_type is coco
                if connection is None:
                    connection = [[16, 14], [14, 12], [17, 15], 
                                  [15, 13], [12, 13], [6, 12], 
                                  [7, 13], [6, 7], [6, 8], 
                                  [7, 9], [8, 10], [9, 11], 
                                  [2, 3], [1, 2], [1, 3], 
                                  [2, 4], [3, 5], [4, 6], [5, 7]]  

            idxs_draw = np.where(kpt[:, 2] != 0)[0]

            if bbox is None:
                bbox = [np.min(kpt[idxs_draw, 0]), np.min(kpt[idxs_draw, 1]),
                        np.max(kpt[idxs_draw, 0]), np.max(kpt[idxs_draw, 1])] # xyxy

            Rfactor = math.sqrt((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / math.sqrt(img.shape[0] * img.shape[1])
            Rpoint = int(min(10, max(Rfactor*10, 4)))
            Rline = int(min(10, max(Rfactor*5, 2)))

            for idx in idxs_draw:
                x, y, v = kpt[idx, :]
                cv2.circle(canvas, (x, y), Rpoint, colors[idx % len(colors)], thickness=-1)

                if v == 2:
                    cv2.rectangle(canvas, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1), colors[idx % len(colors)], 50)
                elif v == 3:
                    cv2.circle(canvas, (x, y), Rpoint+2, colors[idx % len(colors)], thickness=50)

            canvas = np.full((len(canvas), len(canvas[0]), 3), 0, np.uint8)
            cur_canvas = np.full((len(canvas), len(canvas[0]), 3), 0, np.uint8)

            for idx in range(len(connection)):
                idx1, idx2 = connection[idx]
                y1, x1, v1 = kpt[idx1-1]
                y2, x2, v2 = kpt[idx2-1]
                if v1 == 0 or v2 == 0:
                    continue
                mX = (x1+x2)/2.0
                mY = (y1+y2)/2.0
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), 35), int(angle), 0, 360, 1)

                cv2.fillConvexPoly(cur_canvas, polygon, colors[idx % len(colors)])
                canvas = cv2.addWeighted(canvas, 0.1, cur_canvas, 1.0, 0)

            cv2.imwrite("../../../../home/yju/tf/data/%012d_%06d.jpg"%(anns[i]['image_id'], anns[i]['id']), canvas)


def create_answer(coordinate_anno, base_path):
    image_path = 'D:\\GAN\\train\\train2017_mask'
    
    train_images = []
    
    for files in os.walk(image_path):
        train_images = files
        
    train_images = train_images[2]
    
    for p in range(0, len(train_images)):
    #for p in range(0, 5):   
        # 두 이미지를 합치는 코드
        try:
            img_input = plt.imread("D:/GAN/train/train2017/%s"%train_images[p])
            img_target = plt.imread("D:/GAN/train/train2017_mask/%s"%train_images[p])        
        
            img_target = np.expand_dims(img_target, axis=2)
            
            test = img_input | img_target
            
            #temp = img_target.sum(axis=2)
        
            # multiple_result = img_input * temp
        
            # fake_result = multiple_result - temp
        
            # test = (fake_result - img_input) + (fake_result * multiple_result)
        
            # for i in range(len(test)):
            #     for j in range(len(test[i])):
            #         if (test[i][j]) != 0:
            #             test[i][j] = 255
            
            plt.imsave("D:/GAN/train/train2017_comb/%s"%train_images[p], test, cmap='gray')
            #plt.imsave("D:/GAN/train/valB/%s"%train_images[p], test, cmap='gray')
            #print("%s"%train_images[p])
        
        except:
            print("no images")

################################## main ####################################
def main():
    draw_masking()
    #draw_skeleton()
    
if __name__ == '__main__': 
    main()
