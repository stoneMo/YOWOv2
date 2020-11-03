#!/bin/bash
NumEpoch=25
for ((i=1; i<=NumEpoch; i++))
do
    echo $i 
    python ./evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder groundtruths_jhmdb \
                --detfolder ../../jhmdb_detections/detections_$i
done
