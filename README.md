# MaskPose

Enviroment
---
Computing infrastructure  
Python 3.9.7    
CUDA 11.1 relese  
NVIDIA GPU A5000 used  

### Download backbone code from URL
https://github.com/microsoft/human-pose-estimation.pytorch

Installation 
---
1. Follow the Baseline code installation.
2. Change Joints Dataset.py file
3. Change lib/core/function.py
4. Change lib/core/inference.py

Dataset
---  
If you don't need to use other dataset, just download below link.  
Other preprocessing is not requried.  

### Download COCO dataset from URL
https://cocodataset.org/#download

### Whole COCO dataset download link
https://drive.google.com/drive/folders/1qlget_ijZDOxBMrrYR8Tc5yGUdZE67bk?usp=drive_link

```bash
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

Mask image dataset using by data/makedataset.py

How to use
---
CUDA_VISIBLE_DEVICES=0,1,2,3 python pose_estimation/train.py --cfg experiments/coco/resnet152/256x192_d256x3_adam_lr1e-3.yaml
