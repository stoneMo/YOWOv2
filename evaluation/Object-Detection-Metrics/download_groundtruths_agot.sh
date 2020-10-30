#!/bin/bash

# download groundtruths_agot from Google-drive
fileid="1Xwxj9rQClc2yVACrsDzttT9ZuLqjS53L"
filename="groundtruths_agot.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# unzip 
unzip -d ./evaluation/Object-Detection-Metrics/ groundtruths_agot.zip
# mv evaluation/Object-Detection-Metrics/content/gdrive/My\ Drive/project/agot_data/groundtruths_agot evaluation/Object-Detection-Metrics/

# AP: 6.36% (1)
# AP: 1.98% (10)
# AP: 19.72% (11)
# AP: 21.48% (12)
# AP: 1.87% (13)
# AP: 17.87% (14)
# AP: 44.29% (15)
# AP: 0.00% (16)
# AP: 4.39% (17)
# AP: 1.05% (18)
# AP: 9.84% (19)
# AP: 3.89% (2)
# AP: 0.16% (20)
# AP: 0.00% (21)
# AP: 1.67% (22)
# AP: 0.62% (23)
# AP: 6.78% (24)
# AP: 2.17% (3)
# AP: 27.62% (4)
# AP: 8.26% (5)
# AP: 2.96% (6)
# AP: 0.02% (7)
# AP: 0.00% (8)
# AP: 11.12% (9)
# mAP: 8.09%
