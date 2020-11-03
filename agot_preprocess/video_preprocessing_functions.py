
# coding: utf-8

import cv2
import numpy as np
import math   
import matplotlib.pyplot as plt    
import xml.etree.ElementTree as ET
from operator import itemgetter

import os

from sklearn.cluster import KMeans
import random
import seaborn as sns


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


def removeDS(array):
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array


def get_Nclasses_cutoff(path, N):
    '''
    This function is used to get the number of clips contained in the top N classes when sorted by number of clips.
    '''
    labels = os.listdir(path)
    labels = removeDS(labels)
        
    clip_nums = []
    for label in labels:
        label_path = path + label + '/'
        clips = os.listdir(label_path)
        clips = removeDS(clips)
        clip_nums.append(len(clips))
            

    clip_nums.sort(reverse=True)
    cutoff = clip_nums[N-1]
    
    return cutoff


def build_clipref_list(path, list_root, cutoff):
    '''
    This function is used to build training and testing data list we're gonna use. 
    The clip reference (class_group_clip) will be stored in txt file.
    '''
    labels = os.listdir(path)
    labels = removeDS(labels)

    train_refs_full = []
    test_refs_full = []

    for label in labels:
        label_path = path + label + '/'
        clips = os.listdir(label_path)
        clips = removeDS(clips)
        
        num_clips = len(clips)
        
        if num_clips >= cutoff:###HOW TO HANDLE IF MULTIPLE FOLDERS HAVE SAME NUMBER OF FILES
            
            train_num = int(num_clips*train_ratio) # round down, or there might be no data in testing at all

            if clips: # if not empty
                train_refs = clips[0:train_num]
                test_refs = clips[train_num::]

                train_refs = [label + '/' + s for s in train_refs]
                test_refs = [label + '/' + s for s in test_refs]
                
            else:
                train_refs = []
                test_refs = []

            print(label + ':')
            print('total clips: ' + str(len(clips)))
            print('training clips: ' + str(len(train_refs)))
            print('testing clips: ' + str(len(test_refs)))
            print('===========================')

            train_refs_full += train_refs
            test_refs_full += test_refs

    delimiter = '\n'
    train_str = delimiter.join(train_refs_full)
    test_str = delimiter.join(test_refs_full)

    train_path = list_root + 'trainlist01.txt'
    test_path = list_root + 'testlist01.txt'

    file = open(train_path,'w+') 
    file.write(train_str)
    file.close()
    file = open(test_path,'w+')
    file.write(test_str)
    file.close()



def build_labelref_list(path, list_root, cutoff):
    '''
    This function is used to build training and testing data list we're gonna use. 
    The label path (class_group_clip/frameidx.txt) will be stored in txt file.
    '''
    labels = os.listdir(path)

    train_frames_full = []
    test_frames_full = []

    for label in labels:
        if label == '.DS_Store':
            continue
        label_path = path + label + '/'
        clips = os.listdir(label_path)
        num_clips = len(clips)
        
        if num_clips >= cutoff:
            
            train_num = int(num_clips*train_ratio) # round down, or there might be no data in testing at all

            if clips: # if not empty
                train_refs = clips[0:train_num]
                test_refs = clips[train_num::]
                
                train_paths = [label + '/' + s for s in train_refs]
                test_paths = [label + '/' + s for s in test_refs]
                
                for i in range(len(train_refs)):
                    clip_ref = train_refs[i] # class_g_c
                    clip_path = label_path + clip_ref + '/'
                    frames = os.listdir(clip_path)
                    
                    for frame in frames:
                        frame_path = train_paths[i] + '/' + frame
                        train_frames_full.append(frame_path)
                        
                for i in range(len(test_refs)):
                    clip_ref = test_refs[i] # class_g_c
                    clip_path = label_path + clip_ref + '/'
                    frames = os.listdir(clip_path)
                    
                    for frame in frames:
                        frame_path = test_paths[i] + '/' + frame
                        test_frames_full.append(frame_path)

            else:
                train_frames_full = []
                test_frames_full = []

            print(label + ':')
            print('total clips: ' + str(len(clips)))
            print('training clips: ' + str(len(train_refs)))
            print('testing clips: ' + str(len(test_refs)))
            print('===========================')

    delimiter = '\n'
    train_str = delimiter.join(train_frames_full)
    test_str = delimiter.join(test_frames_full)

    train_path = list_root + 'trainlist.txt'
    test_path = list_root + 'testlist.txt'

    file = open(train_path,'w+') 
    file.write(train_str)
    file.close()
    file = open(test_path,'w+')
    file.write(test_str)
    file.close()


def readFile(path):
    with open(path, "rt") as f:
        return f.read()


def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


def gen_groundtruth_folder(testlist_path, label_path, gt_path):
    testlist = readFile(testlist_path).split('\n')
    print("len of testlist:", len(testlist))
    i = 0
    for row in testlist:
        if i % 1000==0:
            print("processing row",i)
        content = readFile(label_path + row)
        gt_file = row.replace("/","_")
        writeFile(gt_path+gt_file, content)
        i += 1


def get_num_clips(path):
    '''
    This function calculates number of clips for each action.
    '''
    labels = os.listdir(path)
    num_clips = []
    for label in labels:
        if label == '.DS_Store':
            continue
        label_path = path + label + '/'
        clips = os.listdir(label_path)
        num_clips.append(len(clips))
    return(num_clips, labels)


def read_labels(path, cutoff):
    '''
    This function is used to get the number of clips contained in the top N classes when sorted by number of clips.
    '''
    labels = os.listdir(path)
    labels = removeDS(labels)
    
    bounding_boxes = []
    i = 0    
    for label in labels:
        print("label =",i)
        label_path = path + label + '/'
        clips = os.listdir(label_path)
        clips = removeDS(clips)

        if len(clips) > cutoff:
            for clip in clips:
                clip_path = label_path + clip
                files = os.listdir(clip_path)
                files = removeDS(files)
                for file in files:
                    file_path = clip_path + '/' + file
                    data = np.loadtxt(file_path)
                    #assuming (x1, y1), (x2, y2)
                    width = (data[3] - data[1])/224
                    height = (data[4] - data[2])/224
                    bounding_boxes.append([width, height])
        i += 1
    
    return bounding_boxes


def jaccard_index(box1, box2):
    num = min(box1[0], box2[0]) * min(box1[1], box2[1])
    den = box1[0]*box1[1] + box2[0]* box2[1] - num
    
    distance = num/den
    
    return distance


def euclidean_dist(box1, box2):
    distance = (box1[0] - box2[0])**2 + (box1[1] - box2[1])**2
    dist = distance ** 0.5 
    
    return dist


def initialize(K):
    cluster_mean = []
    for i in range(K):
        cluster_mean.append([random.random()*0.2, random.random()*0.2])
        
    return cluster_mean
    
    
def find_clusters(bounding_boxes, cluster_mean, K):

    cluster = np.zeros(len(bounding_boxes), dtype = int)
    
    for i in range(len(bounding_boxes)):
        
        min_dist = 1000
        for k in range(K):
            distance = jaccard_index(bounding_boxes[i], cluster_mean[k])
            #distance = euclidean_dist(bounding_boxes[i], cluster_mean[k])
            if distance < min_dist:
                min_dist = distance
                cluster[i] = k

    return cluster
            
            
def new_cluster(bounding_boxes, cluster, old_cluster_mean, K):

    cluster_mean = []
    
    w_sum = np.zeros(K)
    h_sum = np.zeros(K)
    count = np.zeros(K)
    
    for i in range(len(bounding_boxes)):
        w_sum[cluster[i]] += bounding_boxes[i][0]
        h_sum[cluster[i]] += bounding_boxes[i][1]
        count[cluster[i]] += 1
        
    for k in range(K):
        if count[k] != 0:
            cluster_mean.append([w_sum[k]/count[k], h_sum[k]/count[k]])
        else:
            cluster_mean.append([old_cluster_mean[k][0], old_cluster_mean[k][1]])
        
    
    return cluster_mean

