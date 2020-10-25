#!/bin/bash

# download raw agot data from Google-drive
fileid="1mqzVMR5ud1ZzS5h2kvAOhS_UT6D3PKJ3"
filename="agot_raw.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# unzip 
unzip -d ./datasets/ agot_raw.zip
