# Towards Improving Spatiotemporal Action Recognition in Videos

Spatiotemporal action recognition deals with locating and classifying actions in videos. Motivated by the latest state-of-the-art real-time object detector You Only Watch Once (YOWO), we aim to modify its structure to increase action detection precision and reduce computational time. Specifically, we propose four novel approaches in attempts to improve YOWO and address the imbalanced class issue in videos by modifying the loss function. We consider two moderate-sized datasets to apply our modification of YOWO - the popular Joint-annotated Human Motion Data Base (J-HMDB-21) and a private dataset of restaurant video footage provided by a Carnegie Mellon University-based startup, [*Agot.AI*](https://www.agot.ai/). The latter involves fast-moving actions with small objects as well as unbalanced data classes, making the task of action localization more challenging. 

[[Report]](https://www.overleaf.com/read/jdbqkgbfstws)
[[Video]](https://www.youtube.com/watch?v=WIr3QHQWmVs)

## Branches
- Main branch runs the proposed Linknet models.
- Center3D branch runs the proposed Center3D model.
- Baseline branch runs baseline YOWO models and DIYAnchorBox.

## 2D&3D Weights
```
The pretrained weights can be downloaded at ./weights folder.

bash ./weights/download_pretrain.bash
```

## Datasets
### JHMDB-21

```
The JHMDB-21 dataset can be downloaded at ./datasets folder.

bash ./datasets/download_data.bash
```

### Agot-24 dataset (imbalanced)

[[agot-24.zip]](https://drive.google.com/file/d/1xvO5qLBm3Ut0T46R16Cp3wP7I1wHOn4z/view?usp=sharing)  
[[groundtruths_agot.zip]](https://drive.google.com/file/d/1Xwxj9rQClc2yVACrsDzttT9ZuLqjS53L/view?usp=sharing)

```
The Agot-24 dataset can be downloaded at ./datasets folder.

bash ./datasets/download_agot.bash

The corresponding groundtruth file can be downloaded at ./evaluation/Object-Detection-Metrics folder.

bash ./evaluation/Object-Detection-Metrics/download_groundtruths_agot.bash
```

## Running Experiments
### JHMDB-21
```
Running Experiment on JHMDB-21.

bash run_jhmdb-21.sh
``` 

```
Test frame_mAP on JHMDB-21.

bash run_frame_mAP_jhmdb.sh
``` 

### Agot-24 dataset (imbalanced)
```
Running Experiment on Agot dataset.

bash run_agot-24.sh
```

Move the detection folder with the best result to ./evaluation/Object-Detection-Metrics folder.  
Make sure both the detection folder and the groundtruth folder are under the ./evaluation/Object-Detection-Metrics folder.  
```
Test frame_mAP on Agot dataset.

bash run_frame_mAP_agot.sh
``` 

## Acknowledgements

The repository first started as a fork of the [YOWO](https://github.com/wei-tim/YOWO) repository -- we owe a great deal of thanks to the YOWO authors for releasing their codebase.
