# MaskPose

Enviroment
---
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
### Download COCO dataset from URL
https://cocodataset.org/#download

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
