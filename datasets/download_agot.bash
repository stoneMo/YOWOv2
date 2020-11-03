#!/bin/bash

# download agot from Google-drive
fileid="1xvO5qLBm3Ut0T46R16Cp3wP7I1wHOn4z"
filename="agot-24.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# unzip 
unzip -d ./datasets/ agot-24.zip
