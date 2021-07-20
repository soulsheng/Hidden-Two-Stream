#!/usr/bin/env python

import os, sys
import numpy as np
import caffe
import math
import cv2
import scipy.io as sio
#import h5py
from common import *

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

    topN = 3
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

    video_fps = 30.0
    video_size = (640,480)  
    win_size = (2,2)
    
    bReadVideo = True
    
    for line in val_list:
        line_info = line.split(" ")
        input_video_dir_part = line_info[0]
        input_video_dir = os.path.join(FRAME_PATH, input_video_dir_part[:-4])
        input_video_label = int(line_info[1])
        
        if bReadVideo:
            fmt = input_video_dir_part[-3:]
            inVideo = cv2.VideoCapture(FRAME_PATH + '/' + input_video_dir_part)
            inVideoIR = cv2.VideoCapture(input_video_dir + 'ir.'+fmt)
            print('open video ', input_video_dir)
            if inVideo and inVideoIR:
                print('success ')
            else:
                print('failed ')
            
            video_fps = inVideo.get(cv2.CAP_PROP_FPS)
            width = inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_size = ((int)(width/2), (int)(height/2)) 
        
        output_dir = input_video_dir + '/out'
        if not os.path.exists(output_dir):
            os.makedirs( output_dir )
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outVideoFilename = '%d.mp4'%line_id 
        #outVideoFilename = input_video_dir + '/../../../' + input_video_dir_part
        outVideo = cv2.VideoWriter( outVideoFilename, fourcc, video_fps, (video_size[0]*win_size[0],video_size[1]*win_size[1]) )
        print('create video ', outVideoFilename)
        if outVideo:
            print('success ')
        else:
            print('failed ')
        
        if bReadVideo:
            frame_total = int(inVideo.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_totalIR = int(inVideoIR.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_total>frame_totalIR:
                frame_total = frame_totalIR
        else:
            imglist = os.listdir(input_video_dir)
            frame_total = len(imglist)

        nStep = 25
                
        nGroup = frame_total/nStep
        
        labelList = []
        scoreList = []
        correct = 0
        scoreT = 0.8
        time_slice = 5 #0 for all 
        print( 'nGroup = frame_total/nStep, %d = %d/%d'%(nGroup, frame_total, nStep) )
        for group in range(nGroup):
            if group*nStep >time_slice*video_fps: break
            print( '\n' )
            print( 'group id = ', group)
            start_frame = group*nStep
            num_frames = nStep 
                    
            frameList = []
            frameListIR = []
            for i in range(num_frames):
                id = start_frame+i
                img_file = os.path.join(input_video_dir, 'image_{0:04d}.jpg'.format(id))
                if bReadVideo:
                    ret,img = inVideo.read()
                    if ret == False:
                        break  
                        
                    ret,imgIR = inVideoIR.read()                    
                    if ret == False:
                        break                       
                    frameListIR.append(imgIR)
                    
                else:
                    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print('failed to open ', id)
                        continue
                    frameList.append(img)
                
                frameList.append(img) 
                
            rect_merge,rectAll = mergeRect(frameList, face_net)
            rect_scale = scaleRect(rect_merge, frameList[0].shape, 256, 340)
            
            spatial_prediction = HiddenTemporalPrediction(rect_scale, 
                    frameList,
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
            
            bCorrect, scoreTop0 = isCorrect(avg_spatial_pred, input_video_label, topN, scoreT)
            scoreList.append(scoreTop0) 
            
            if bCorrect:
                correct += 1
            writeFrames(frameList, rectAll, bCorrect, start_frame, video_size, frameListIR, output_dir, outVideo)
            
            if cv2.waitKey(10) == 27: # 27 = 'esc'
                break
                
        line_id += 1
        outVideo.release()      
        inVideo.release()
        print('write video ', outVideoFilename)
        print('labelList =', labelList)
        print('scoreList =', scoreList)
        print('correct = ', correct)
        print( "prediction accuracy is %4.4f" % (float(correct)/nGroup))

    
if __name__ == "__main__":
    main()
