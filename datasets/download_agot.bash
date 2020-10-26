#!/bin/bash

# download jhmdb from Google-drive
fileid="1ZqFneqlFuHiqTSk2npDZfsv28KBd7XjJ"
filename="agot-24.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# unzip 
unzip -d ./datasets/ agot.zip

