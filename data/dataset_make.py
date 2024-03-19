# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:35:28 2022

@author: labpc
"""

import json

def bring_json(base_path, coordinate):  
        
    # with open(base_path + 'data\\OCHuman\\annot\\ochuman_coco_format_all_range_0.00_1.00.json' ,  'r') as f:
    #     coordinate = json.load(f)
    with open(base_path + 'data\\coco\\annotations\\person_keypoints_train2017.json' ,  'r') as f:
        coordinate = json.load(f) 
    
    coordinate = coordinate['annotations']
    
    temp = [] 
    for i in range(0, len(coordinate)):
        if coordinate[i]['num_keypoints'] > 0:
            temp.append(coordinate[i])
    
    coordinate = temp

    return coordinate

        
############################## 이미지별 keypoint 모으기 ###############################
import matplotlib.pyplot as plt

def info_combination(coordinate):    
   
    for i in range(0, len(coordinate['annotations'])):
        for k in range(0, len(coordinate['images'])):
            if coordinate['annotations'][i]['image_id'] == coordinate['images'][k]['id'] :
                coordinate['annotations'][i]['file_name'] = coordinate['images'][k]['file_name']
    
    return coordinate

############################## 폴더 경로 없으면 새로 생성 ###############################
import os

def create_path(folder_date):   
    if not os.path.exists('D:/GAN/output/{}'.format(folder_date)):
        os.makedirs('D:/GAN/output/{}'.format(folder_date))

################################## 현재 시간 추출 #####################################        
from datetime import datetime
def cal_time(return_time):
    #pred 값 저장
    now = datetime.now()
    date = now.date()
    hour = now.hour
    minute = now.minute
    
    now_time = str(date) + "_" + str(hour) + "_" + str(minute)
    
    return now_time

################################## circle drawing ####################################
def show_img(json_pred, base_path, folder_date, img_count=10, circle_size=50):
    color = ['cornflowerblue', 'oldlace', 'yellow', 'black', 'red', 'blue', 'pink', 'orange', 'purple', 'lightcyan', 'gray', 'bisque', 'lightgreen', 'violet'
             , 'peru', 'rosybrown', 'gold', 'ivory', 'azure', 'chocolate', 'darkolivegreen', 'indigo', 'green'
             , 'darkgrey', 'grey', 'tan', 'white', 'lawngreen']
    
    start_img = json_pred[0]['image_id']
    color_id = 0
    detected_person = 0
    
    img_count = 10 
    circle_size = 30
    
    for i in range(0, img_count):
        
        temp_x = []
        temp_y = []
        
        image = plt.imread(base_path + "data\\OCHuman\\images\\%s"%json_pred[i]['file_name'])
        
        plt.imshow(image)
        
        ax = plt.gca()
        
        for j in range(0, len(json_pred[i]['keypoints']), 3):
            circle = plt.Circle((json_pred[i]['keypoints'][j], json_pred[i]['keypoints'][j+1]), circle_size, color=color[0])
            ax.add_patch(circle)
            temp_x.append(json_pred[i]['keypoints'][j])
            temp_y.append(json_pred[i]['keypoints'][j+1])
            
        plt.plot(temp_x[0:5], temp_y[0:5], c = 'g', linewidth = 5)
        plt.plot(temp_x[2:4], temp_y[2:4], c = 'g', linewidth = 5)
            
        plt.show()   
       
################################## draw skeleton ####################################
import numpy as np
import math
import cv2
from PIL import Image

def draw_skeleton(json_pred, base_path, connection=None, colors=None, bbox=None):
    
    coco = COCO(base_path + 'data\\coco\\annotations\\person_keypoints_train2017.json')
    coco_keys = coco.imgs.keys()
    
    #for j in range(start, len(json_pred)):
    for k, j in enumerate(coco_keys):    
        #print(j)
        if k > 15000:
            break
        
        img = coco.imgs[j]
        
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        
        for i in range(len(anns)): 
            kpt = anns[i]['keypoints']
            kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
            npart = kpt.shape[0]
    
    #for i in range(0, len(json_pred)):    
        #kpt = json_pred[i]['keypoints']
        # kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
        # npart = kpt.shape[0]
        
        #img = plt.imread(base_path + "data\\OCHuman\\images\\%s"%json_pred[i]['file_name'])
            img = plt.imread(base_path + "data\\coco\\images\\train2017\\%012d.jpg"%anns[i]['image_id'])
            canvas = img.copy()
            
            if npart==17: # ochuman, data_type is coco
                part_names = ['nose', 
                              'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                              'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                              'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']         
                visible_map = {2: 'vis', 
                               1: 'not_vis', 
                               0: 'missing'}
                map_visible = {value: key for key, value in visible_map.items()}
                if connection is None:
                    connection = [[16, 14], [14, 12], [17, 15], 
                                  [15, 13], [12, 13], [6, 12], 
                                  [7, 13], [6, 7], [6, 8], 
                                  [7, 9], [8, 10], [9, 11], 
                                  [2, 3], [1, 2], [1, 3], 
                                  [2, 4], [3, 5], [4, 6], [5, 7]]            
            
            #원래 무지개빛 컬러 코드
            # if colors is None:
            #     colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], 
            #              [255, 255, 0], [170, 255, 0], [85, 255, 0], 
            #              [0, 255, 0], [0, 255, 85], [0, 255, 170], 
            #              [0, 255, 255], [0, 170, 255], [0, 85, 255], 
            #              [0, 0, 255], [85, 0, 255], [170, 0, 255],
            #              [255, 0, 255], [255, 0, 170], [255, 0, 85]]
                
            if colors is None:
                colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], 
                         [255, 255, 255], [255, 255, 255], [255, 255, 255], 
                         [255, 255, 255], [255, 255, 255], [255, 255, 255], 
                         [255, 255, 255], [255, 255, 255], [255, 255, 255], 
                         [255, 255, 255], [255, 255, 255], [255, 255, 255],
                         [255, 255, 255], [255, 255, 255], [255, 255, 255]]
            
            elif type(colors[0]) not in [list, tuple]:
                colors = [colors]
            
            idxs_draw = np.where(kpt[:, 2] != map_visible['missing'])[0]
            # if len(idxs_draw)==0:
            #     return img
            
            if bbox is None:
                bbox = [np.min(kpt[idxs_draw, 0]), np.min(kpt[idxs_draw, 1]),
                        np.max(kpt[idxs_draw, 0]), np.max(kpt[idxs_draw, 1])] # xyxy
            
            Rfactor = math.sqrt((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / math.sqrt(img.shape[0] * img.shape[1])
            Rpoint = int(min(10, max(Rfactor*10, 4)))
            Rline = int(min(10, max(Rfactor*5, 2)))
            #print (Rfactor, Rpoint, Rline)
            
            for idx in idxs_draw:
                x, y, v = kpt[idx, :]
                cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
                
                if v==2:
                    cv2.rectangle(canvas, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1), colors[idx%len(colors)], 50)
                elif v==3:
                    cv2.circle(canvas, (x, y), Rpoint+2, colors[idx%len(colors)], thickness=50)
                    
            canvas = np.full((len(canvas), len(canvas[0]), 3), 0, np.uint8)
            cur_canvas = np.full((len(canvas), len(canvas[0]), 3), 0, np.uint8)
            
            for idx in range(len(connection)):
                idx1, idx2 = connection[idx]
                y1, x1, v1 = kpt[idx1-1]
                y2, x2, v2 = kpt[idx2-1]
                if v1 == map_visible['missing'] or v2 == map_visible['missing']:
                    continue
                mX = (x1+x2)/2.0
                mY = (y1+y2)/2.0
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
                # polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), Rline), int(angle), 0, 360, 1)
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), 35), int(angle), 0, 360, 1)
                # cur_canvas = canvas.copy()           
                
                cv2.fillConvexPoly(cur_canvas, polygon, colors[idx%len(colors)])
                canvas = cv2.addWeighted(canvas, 0.1, cur_canvas, 1.0, 0)            
    
                #cv2.imwrite("D:/GAN/train/train/%s_%04d.jpg"%(json_pred[i]['file_name'].split(".")[0], json_pred[i]['id']), canvas)
                cv2.imwrite("D:/GAN/train/train2017/%012d_%06d.jpg"%(anns[i]['image_id'], anns[i]['id']), canvas)
        
        # cv2.imshow('dst', canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
################################## draw masking ####################################
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def draw_masking(json_pred, base_path, start, end):
    # coco = COCO(base_path + 'data\\OCHuman\\annot\\ochuman_coco_format_all_range_0.00_1.00.json')
    # img_dir = base_path + 'data\\OCHuman\\images'
    
    coco = COCO(base_path + 'data\\coco\\annotations\\person_keypoints_train2017.json')
    img_dir = base_path + 'data\\coco\\images\\train2017'
    
    coco_keys = coco.imgs.keys()
    
    #for j in range(start, len(json_pred)):
    for k, j in enumerate(coco_keys):    
        #print(j)
        if k > 20000:
            break
        img = coco.imgs[j]
        
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)    
        
        #mask = coco.annToMask(anns[0])
        
        temp_id = []
        
        for i in range(len(json_pred)):
            if json_pred[i]['image_id'] == img['id']:
                temp_id.append(json_pred[i])
        
        for i in range(len(anns)):            
            # mask += coco.annToMask(anns[i])  #전체 마스크를 그려주는 코드          
            mask = coco.annToMask(anns[i])
        
            temp = Image.fromarray(mask*255)          

            #temp.save("D:/GAN/train/trainB/%s_%04d.jpg"%(img['file_name'].split(".")[0], temp_id[i]['id']), cmap='gray')
            temp.save("D:/GAN/train/train2017_mask/%s_%04d.jpg"%(anns[i]['image_id'].split(".")[0], anns[i]['id']), cmap='gray')

                # plt.figure(dpi=250)
                #plt.Figure(figsize= (temp_id[0]['segmentation']['size'][0], temp_id[0]['segmentation']['size'][1]))
                # ax = plt.gca()
                # ax.set_axis_off()            
                # plt.tight_layout()
                # plt.imshow(mask, interpolation='nearest')
                # plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
                # plt.savefig("D:/GAN/train/%s_%04d.jpg"%(img['file_name'].split(".")[0], temp_id[i]['id']), bbox_inches='tight', pad_inches=0)
            #print("D:/GAN/trainB/%s_%04d.jpg"%(img['file_name'].split(".")[0], temp_id[i]['id']))

from PIL import ImageOps            
def rainbow_to_two():    
    image_path = 'D:\\GAN\\train\\trainB\\'
    
    trainB_iamges = []
    
    for files in os.walk(image_path):
        trainB_iamges = files
        
    trainB_iamges = trainB_iamges[2]
    
    original = plt.imread('D:\\MIPNet-main\\data\\OCHuman\\images\\000001.jpg')
    two_img = plt.imread('D:\\GAN\\train\\trainA\\000001_7071.jpg')
    
    for l in range(0, 1):
    #for l in range(0, len(trainB_iamges)):        
        rainbow = plt.imread(image_path + "%s"%trainB_iamges[l])
        
        for i in range(len(rainbow)):
            for j in range(0, len(rainbow[i])):
                if sum(rainbow[i][j]) == 0:
                    rainbow[i][j] = np.array([68, 0, 83])
                else:
                    rainbow[i][j] = np.array([253,230,36])
                
        image = Image.fromarray(rainbow)
        image.save("D:\\GAN\\train\\trainB_\\%s"%trainB_iamges[l])

    
def check(coordinate):    
    for i in range(0, len(coordinate['images'])):
        if coordinate['images'][i]['file_name'] == '003799.jpg':
            print(coordinate['images'][i]['id'])
            
    for i in range(0, len(coordinate['annotations'])):
        if coordinate['annotations'][i]['image_id'] == 1:
            print(coordinate['annotations'][i])
            first_image_box = coordinate['annotations'][i]['bbox']
    
    img = Image.open('D:\\MIPNet-main\\data\\OCHuman\\images\\000001.jpg')
    crop = ImageOps.crop(img, (first_image_box[0], first_image_box[1], img.size[0]-first_image_box[2]-first_image_box[0], img.size[1]-first_image_box[3]-first_image_box[1]))
    
    img = Image.open('D:\\GAN\\train_old\\003799_0001.jpg')
    crop = ImageOps.crop(img, (first_image_box[0]*2, first_image_box[1]*2, first_image_box[2]*2-first_image_box[2]*2, first_image_box[3]*2-first_image_box[3]*2))
    
    crop.show()

def make_color():
    image_path = 'D:\\GAN\\train\\train2017'
    
    train_images = []
    
    for files in os.walk(image_path):
        train_images = files
        
    train_images = train_images[2]
    
    for p in range(79286, len(train_images)):        
        img_color = plt.imread("D:/MIPNet-main/data/coco/images/train2017/%s.jpg"%train_images[p].split("_")[0])        
        img_mask = plt.imread("D:/GAN/train/train2017/%s"%train_images[p])       
        
        if os.path.isfile("D:/GAN/train/train2017_color/%s"%train_images[p]):
            if p % 1000 == 0:
                print(p)
            pass
        else:            
            try:
                img_con = (img_color & img_mask)
            except:
                print(p)
                img_color = np.expand_dims(img_color, axis=2)
                img_con = (img_color & img_mask)
                
            #plt.imshow(img_con)
            plt.imsave("D:/GAN/train/train2017_color/%s"%train_images[p], img_con)

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
    # save_time = ''    
    # save_time = cal_time(save_time)
    
    # create_path(save_time)    
    
    base_path = 'D:\\MIPNet-main\\'
    
    coordinate = []    
    coordinate = bring_json(base_path, coordinate)
    
    coordinate = info_combination(coordinate)
    
    coordinate_anno = coordinate['annotations']
    
    # show_img(coordinate_anno, base_path, save_time, 5, 30)
    
    start_num = 1
    end_num = 0    
    
    # draw_masking(coordinate_anno, base_path, start_num, end_num)
    
    #draw_skeleton(coordinate_anno, base_path)
    
if __name__ == '__main__': 
    main()