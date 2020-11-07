# Spatiotemporal Action Recognition in Videos

Shentong Mo, Xiaoqing Tan, Jingfei Xia, Pinxu Ren.

## Branches
- Main branch runs customized YOWO-linknet models.
- Baseline branch runs baseline YOWO models.

## Project report 
https://www.overleaf.com/read/jdbqkgbfstws

## 2D&3D Weights
```
The pretrained weights can be downloaded at ./weights folder.

bash ./weights/download_pretrain.bash
```

## Dataset 
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

## Running Experiment
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

```
Test frame_mAP on Agot dataset.

bash run_frame_mAP_agot.sh
``` 

## Reference paper

YOWO: https://github.com/wei-tim/YOWO
