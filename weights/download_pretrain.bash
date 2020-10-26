#!/bin/bash
# https://drive.google.com/file/d/1uoDK_wng3AgqkemmstQoRLQfbQmqHdC2/view?usp=sharing

wget http://pjreddie.com/media/files/yolo.weights

# fileid="1cULocPe5YvPGWU4tV5t6fC9rJdGLfkWe"
# filename="resnext-101-kinetics.pth"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# fileid="1GWP0bAff6H6cE85J6Dz52in6JGv7QZ_u"
# filename="resnext-101-kinetics-hmdb51_split1.pth"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

fileid="1id0l08FAYwFykhVo0VId8s6Ss1a162WI"
filename="resnet-18-kinetics.pth"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# fileid="1kDEtgOL9-hbhEono5a29NeFVh_MMvaet"
# filename="resnet-18-kinetics-ucf101_split1.pth"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
