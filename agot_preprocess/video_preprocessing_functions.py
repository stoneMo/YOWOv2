
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math   
import matplotlib.pyplot as plt    
import xml.etree.ElementTree as ET
from operator import itemgetter

import os


# In[2]:


def get_labels(filePath):
    '''
    DEFINITION : This function creates a list of all labels/actions in our dataset 
    
    INPUT : A list that contains the file paths of all .xml labeling files
    OUTPUT : A list of 'labels/actions' in the dataset
    '''
    all_labels = []
    
    for path in filePath:
        root = ET.parse(path).getroot()

        meta = root.find('meta')
        task = meta.find('task')
        labels = task.find('labels')
        
        for label in labels.findall('label'):
            label_name = label.find('name').text
            label_name = label_name.replace('/',' or ')
            
            if label_name not in all_labels:
                all_labels.append(label_name)
            

    return all_labels
        
#all_labels = get_labels(filePath)


# In[125]:


def mkdir_label(labels, root, labels_root, images_root ):
    '''
    DEFINITION : This function creates a UCF101-24 style directory structure
    
    INPUT : A list of labels which we want to use during training
    OUTPUT :
    
    '''
    
    #Create a folder called ucf
    os.mkdir(root)
    os.mkdir(labels_root)
    os.mkdir(images_root)
    
    for label_name in labels:
        label_path = labels_root + label_name
        image_path = images_root + label_name
        
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
            
#mkdir_label(all_labels, root, labels_root, images_root)


# In[3]:


def get_clips(filePath):
    '''
    DEFINITION : This function creates a dictionary of all clips for the actions/labels in our dataset 
    
    INPUT : A list that contains the file paths of all .xml labeling files
    OUTPUT : Number of total videos in dataset
            A sorted dictionary of number of clips per action/label. Dictionary is sorted in decreasing order
    '''

    total_videos = 0
    all_clips = {}
    for path in filePath:
        root = ET.parse(path).getroot()

        for track in root.findall('track'):
            total_videos += 1
            label = track.get('label')
            label = label.replace('/',' or ')
        
            if label in all_clips:
                all_clips[label] += 1
            else:
                all_clips[label] = 1
                
    all_clips = dict(sorted(all_clips.items(), key=itemgetter(1), reverse = True))
    return [total_videos, all_clips]

#total_videos, all_clips = get_clips(filePath)


# In[4]:


def generate_ids(all_clips):
    '''
    DEFINITION : This function creates id's for labels/actions

    INPUT : A sorted dictionary of labels/actions with number of clips
    OUTPUT :
    '''
    label_ids = {}
    for i, label in enumerate(all_clips):
        label_ids[label] = i+1
        
    return label_ids
        
#label_ids = generate_ids(all_clips)


# In[2]:


def create_labels(filePath, labels_root, label_ids, all_labels, x_scale, y_scale):
    
    '''
    x_scale: scaling factor along the x-direction 
    y_scale: scaling factor along the y-direction
    
    '''
    
    #This variable does the same job as label_file_counter
    frames_for_video = []
    label_files = []
    
    
    for p, path in enumerate(filePath):
        root = ET.parse(path).getroot()
        meta = root.find('meta')
        task = meta.find('task')
        ids = int(task.find('id').text)
        
        label_file_counter = dict.fromkeys(all_labels, 0)#keeps count of how many clips each label has
        
        
        frames_list = []
        for track in root.findall('track'):
            if p == 0 and track.get('id') == '11':
                continue
                
            label = track.get('label')
            label = label.replace('/',' or ')
            
            label_file_counter[label] += 1
            
            #create new clip folder
            label_file_path = labels_root + label + '/' + 'g_' + f'{ids:05}' + '_c_' + f'{label_file_counter[label]:05}' + '/'
            os.mkdir(label_file_path)
            
            frames = []
            for i, box in enumerate(track.findall('box')):
                frames.append(box.get('frame'))
                xtl = str(round( float(box.get('xtl'))*x_scale, 3))
                ytl = str(round( float(box.get('ytl'))*y_scale, 3))
                xbr = str(round( float(box.get('xbr'))*x_scale, 3))
                ybr = str(round( float(box.get('ybr'))*y_scale, 3))
                
                box_info = str(label_ids[label]) + ' ' + xtl + ' ' + ytl + ' ' + xbr + ' ' + ybr
                
                label_file_name = label_file_path + f'{i+1:05}' + '.txt'
                f = open(label_file_name,'w+')
                f.write(box_info)
                f.close()
            
            end_frame = box.get('frame')
            frames_list.append([label, frames])
        
        frames_for_video.append(frames_list)
        label_files.append(label_file_counter)
            
    print(sum(label_files[0].values()) + sum(label_files[1].values()))
    return [frames_for_video, label_file_counter]
        
        
#frames_for_video, label_file_counter = create_labels(filePath, labels_root, label_ids, all_labels,  x_scale, y_scale)


# In[1]:


def video_to_frame(filePath, videoPath, images_root, frames_for_video, all_labels, target_size_x, target_size_y):
    '''
    This method is not naive. Takes a lot of time. But the only way to do the task on a machine without a GPU
    '''
    
    at_video_number = 0
    image_files = []

    
    for i, path in enumerate(videoPath):
        #if i == 0:
        #    continue
        root = ET.parse(filePath[i]).getroot()
        meta = root.find('meta')
        task = meta.find('task')
        ids = int(task.find('id').text)
        
        video = cv2.VideoCapture(path)
        frameRate = video.get(5)
        
        clip_counter = dict.fromkeys(all_labels, 0)
        
        for label, frames in frames_for_video[i]:
            at_video_number += 1
            print('video number = ', at_video_number)
            clip_counter[label] += 1
            
            #create new clip folder
            clip_path = images_root + label + '/' + 'g_' + f'{ids:05}' + '_c_' + f'{clip_counter[label]:05}' + '/'
            os.mkdir(clip_path)
            print(clip_path)
            
            if video.isOpened(): # if video open successfully (if False all the time, check ffmpeg)
                for t, idx in enumerate(frames):
                    frame_idx = int(idx)
                
                    video.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
                    ret, frame = video.read()
                
                    if ret: # if read frame successfully
                        image_name = clip_path + f'{t+1:05}' + '.jpg'
                        resized = cv2.resize(frame, (target_size_x, target_size_y), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(image_name, resized)
                        
            
        image_files.append(clip_counter)
    
    print(sum(image_files[0].values()) + sum(image_files[1].values()))

        
        
#video_to_frame(filePath, videoPath, images_root, frames_for_video, all_labels, target_size_x, target_size_y)

