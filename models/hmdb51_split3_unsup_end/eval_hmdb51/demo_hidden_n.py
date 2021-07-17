#!/usr/bin/env python

import os, sys
import numpy as np
import caffe
import math
import cv2
import scipy.io as sio
#import h5py

from HiddenTemporalPrediction import HiddenTemporalPrediction

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():

    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # spatial prediction
    model_def_file = '../stack_motionnet_vgg16_deploy.prototxt'
    model_file = '../logs_end/hmdb51_split3_vgg16_hidden.caffemodel'
    FRAME_PATH = "G:/action/hmdb51-sub4-smoke-alike/"
    spatial_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    face_net = net = cv2.dnn.readNetFromCaffe("../face_deploy.prototxt", "../face_res10_300x300_ssd_iter_140000.caffemodel")

    val_file = "./testlistdiy.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    topN = 5
    start_frame = 0
    num_categories = 51
    feature_layer = 'fc8_vgg16'
    spatial_mean_file = './rgb_mean.mat'
    dims = (len(val_list), num_categories)
    predict_results_before = np.zeros(shape=dims, dtype=np.float64)
    predict_results = np.zeros(shape=dims, dtype=np.float64)

    correct = 0
    line_id = 0
    spatial_results_before = {}
    spatial_results = {}

    for line in val_list:
        line_info = line.split(" ")
        input_video_dir_part = line_info[0]
        input_video_dir = os.path.join(FRAME_PATH, input_video_dir_part[:-4])
        input_video_label = int(line_info[1])
 

        imglist = os.listdir(input_video_dir)
        frame_total = len(imglist)

        nStep = 25
                
        nGroup = frame_total/nStep
        
        labelList = []
        scoreList = []
        correct = 0
        scoreT = 0.8
        print( 'nGroup = frame_total/nStep, %d = %d/%d'%(nGroup, frame_total, nStep) )
        for group in range(nGroup):
            print( '\n' )
            print( 'group id = ', group)
            start_frame = group*nStep
            num_frames = nStep 
            spatial_prediction = HiddenTemporalPrediction(face_net, 
                    input_video_dir,
                    spatial_mean_file,
                    spatial_net,
                    num_categories,
                    feature_layer,
                    start_frame,
                    num_frames)
            avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
            avg_spatial_pred = np.asarray(softmax(avg_spatial_pred_fc8))
            predict_label = np.argmax(avg_spatial_pred)

            labelList.append(predict_label)
            print( input_video_dir)
            print( input_video_label-1, predict_label)
            
            
            #print( avg_spatial_pred )
            ids_sort = avg_spatial_pred.argsort()
            ids_topN = ids_sort[:-(topN+1):-1]
            print( ids_topN )
            print( avg_spatial_pred[ids_topN] )
            scoreList.append(avg_spatial_pred[ids_topN[0]]) 
            
            ids_topN = ids_topN.tolist()
            if input_video_label-1 in ids_topN:
                index = ids_topN.index( input_video_label-1 )
                if index == 0:
                     print('top0')
                     correct += 1
                elif index>0:
                    iStop = 0
                    for i in range(index):
                        score = avg_spatial_pred[ ids_topN[i] ]
                        if score > scoreT:
                            break
                        else:
                            iStop+=1
                    #print('iStop=%d, index=%d'%(iStop, index) )
                    if iStop==index:
                        correct += 1
                        print('top%d'%index, ids_topN)
                
        print('labelList =', labelList)
        print('scoreList =', scoreList)
        print('correct = ', correct)
        print( "prediction accuracy is %4.4f" % (float(correct)/nGroup))

if __name__ == "__main__":
    main()
