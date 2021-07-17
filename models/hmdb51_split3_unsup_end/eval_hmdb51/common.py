
import os, sys
import numpy as np
import caffe
import math
import cv2
import scipy.io as sio
    
def scaleRect(rect, size, height = 256, width = 340 ):
    rectNew = rect
    
    xCenter = int( (rect[0]+rect[2]+1)*0.5 )
    yCenter = int( (rect[1]+rect[3]+1)*0.5 )
    
    rectNew[0] = xCenter-width/2
    if rectNew[0]<0: rectNew[0]=0
    
    rectNew[2] = xCenter+width/2
    if rectNew[2]>size[1]: rectNew[2]=size[1]
       
    rectNew[1] = yCenter-height/2
    if rectNew[1]<0: rectNew[1]=0
    
    rectNew[3] = yCenter+height/2
    if rectNew[3]>size[0]: rectNew[3]=size[0]
    
    return rectNew
    
def detectFace(net, frame):
	box = np.zeros(4)
	(h, w) = frame.shape[:2]
	#print( frame.shape[:2] )
	blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0),False,True) 
	# pass the blob through the network and obtain the detections
	net.setInput(blob)
	detections = net.forward()
 
	confidence = detections[0, 0, 0, 2] 
	if confidence > 0.5: 
		box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h]) 
        
	return box.astype("int")

def mergeRect(frameList, face_net):
 
    _xMin=10000 
    _yMin=10000 
    _xMax=0 
    _yMax=0
    rectList = []
    for i in range( len(frameList) ):
        #print('img[%d]'%i, frameList[i].shape)
        face_rect = detectFace(face_net, frameList[i])
        if _xMin>face_rect[0]: _xMin=face_rect[0]
        if _yMin>face_rect[1]: _yMin=face_rect[1]
        if _xMax<face_rect[2]: _xMax=face_rect[2]
        if _yMax<face_rect[3]: _yMax=face_rect[3]   
        rectList.append( face_rect )
        
    return  [_xMin, _yMin, _xMax, _yMax], rectList


def isCorrect(avg_spatial_pred, input_video_label, topN, scoreT):

    ids_sort = avg_spatial_pred.argsort()
    ids_topN = ids_sort[:-(topN+1):-1]
    print( ids_topN )
    print( avg_spatial_pred[ids_topN] )
    
    bCorrect = False
    
    ids_topN = ids_topN.tolist()
    if input_video_label-1 in ids_topN:
        index = ids_topN.index( input_video_label-1 )
        if index == 0:
             print('top0')
             bCorrect = True
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
                bCorrect = True
                print('top%d'%index, ids_topN)
                
    return bCorrect, avg_spatial_pred[ids_topN[0]]
    

def writeFrames(frameList, rectAll, bCorrect, start_frame, video_size, outPath='./', outVideo=None):
    
    for i in range( len(frameList) ):
        img = frameList[i].copy()
        rect = rectAll[i]
        if bCorrect:
            cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (255, 0, 0), 2 ) 
            cv2.putText(img, 'smoking', (rect[0],rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
            (255, 255, 255), 2) 
        
        #cv2.imshow('out', img)
        #cv2.waitKey(0)
        cv2.imwrite(outPath+'/%04d.jpg'%(i+start_frame), img)
        if outVideo:
            img_resize_in = cv2.resize(frameList[i], video_size) 
            img_resize_out = cv2.resize(img, video_size) 
            img_resize = np.hstack( (img_resize_in, img_resize_out) )
            outVideo.write(img_resize)
        
    return
    