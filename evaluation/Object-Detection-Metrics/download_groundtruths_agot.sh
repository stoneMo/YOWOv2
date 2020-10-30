#!/bin/bash

# download groundtruths_agot from Google-drive
fileid="1Xwxj9rQClc2yVACrsDzttT9ZuLqjS53L"
filename="groundtruths_agot.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# unzip 
unzip -d ./evaluation/Object-Detection-Metrics/ groundtruths_agot.zip
# mv evaluation/Object-Detection-Metrics/content/gdrive/My\ Drive/project/agot_data/groundtruths_agot evaluation/Object-Detection-Metrics/
