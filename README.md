# One-Click Dense Pose

```!git clone https://github.com/simsim314/one-click-dense-pose.git```

## Overview
Logical continuation of the [original repository](https://github.com/Pawandeep-prog/one-click-dense-pose)

Added detector.py and modified helper.py in utils. 
## Installation

[start.ipynb](start.ipynb)

## Run

0. sh ./download-models.sh

1. Old method for videos
    
   ```bash
   python convert.py --input PATH_TO_INPUT_VIDEO --output OUTPUT_VIDEO_PATH
   ```

2. New method on folder images

   it creates several files per input file UV map and the original segmentation
   
   ```bash
   python detect.py -i INPUT_FOLDER -o OUTPUT_FOLDER
   ```

3. Newer method will generate .npz of each image file detections (no gradients pure organ detections) so you could analyze it further the pipeline and not have a problem of compression.
It also keeps the same format and creates the same directory structure in output traversing in all subfolders not only directly under.

   ```bash
   python detect_np.py -i INPUT_FOLDER -o OUTPUT_FOLDER
   ```
   much more useful for data processig as part of larger projects. 
   *Note: It run in parallel to optimize resources. 
   
## Contributions Welcome

Welcome to contribute here or to the original or fork and do whatever you want. 

## Notes

[Check this out](https://github.com/davidleejy/DensePose/tree/speedup/notebooks)

(https://www.kaggle.com/code/hamditarek/sartorius-cell-instance-segmentation-detectron2)


MODEL_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
WEIGHT_PATH = '../input/r101fpn400/model_final_f96b26.pkl'


(https://github.com/facebookresearch/detectron2/issues/4737#issuecomment-1383064054)

----

## Train

(https://blog.roboflow.com/how-to-train-detectron2/)

Detectron other models:

(https://www.youtube.com/watch?v=sYEZSWJXQZs)

[Kaggle Examples](https://www.kaggle.com/code/maartenvandevelde/object-detection-with-detectron2-pytorch)

[Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
