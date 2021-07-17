import glob
import os, sys
import numpy as np
import math, time
import cv2
import scipy.io as sio

def HiddenTemporalPrediction(rect_scale, 
        frameList,
        mean_file,
        net,
        num_categories,
        feature_layer,
        start_frame=0,
        duration=0,
        num_samples=4,
        stacked_frames=11
        ):
    

    # selection
    step = int(math.floor((duration-stacked_frames+1)/num_samples))
    dims = (256,340,stacked_frames*3,num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)
    #print('duration=%d, stacked_frames=%d, num_samples=%d, step=%d'%(duration, stacked_frames, num_samples, step) )
    for i in range(num_samples):
        #print( 'sample id = ', i)
        stacked_list = []
        stacked_flip_list = []
        face_rect = np.zeros(4, np.int)
        for j in range(stacked_frames):
            #img_file = os.path.join(vid_name, 'image_{0:04d}.jpg'.format(i*step+j+1 + start_frame))
            #img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = frameList[i*step+j+1].copy()
            #print( i*step+j+1 )
            if 1:
                img = img[rect_scale[1]:rect_scale[3], rect_scale[0]:rect_scale[2]]     
                
                if 0:#j==0:          
                    print( 'rect scale ', rect_scale)
                    print( 'img size', img.shape )
                    cv2.imshow('img rect', img)
                    cv2.waitKey(0)  
            img = cv2.resize(img, dims[1::-1])
            stacked_list.append(img)
            stacked_flip_list.append(img[:,::-1,:])
        stacked_img = np.concatenate(stacked_list, axis=2)
        stacked_flip_img = np.concatenate(stacked_flip_list, axis=2)
        rgb[:,:,:,i] = stacked_img
        rgb_flip[:,:,:,i] = stacked_flip_img

    # crop
    rgb_1 = rgb[:224, :224, :,:]
    rgb_2 = rgb[:224, -224:, :,:]
    rgb_3 = rgb[16:240, 60:284, :,:]
    rgb_4 = rgb[-224:, :224, :,:]
    rgb_5 = rgb[-224:, -224:, :,:]
    rgb_f_1 = rgb_flip[:224, :224, :,:]
    rgb_f_2 = rgb_flip[:224, -224:, :,:]
    rgb_f_3 = rgb_flip[16:240, 60:284, :,:]
    rgb_f_4 = rgb_flip[-224:, :224, :,:]
    rgb_f_5 = rgb_flip[-224:, -224:, :,:]

    rgb = np.concatenate((rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5), axis=3)

    # substract mean and divide by 255 because of flow estimation
    d = sio.loadmat(mean_file)
    image_mean = d['image_mean']
    image_mean = np.tile(image_mean, (1,1,stacked_frames))
    rgb = rgb - np.tile(image_mean[...,np.newaxis], (1, 1, 1, rgb.shape[3]))
    rgb = np.transpose(rgb, (1,0,2,3))
    rgb = rgb / 255

    #print( 'rgb.shape ', rgb.shape )
    
    # test
    batch_size = 20
    prediction = np.zeros((num_categories,rgb.shape[3]))
    num_batches = int(math.ceil(float(rgb.shape[3])/batch_size))


    for bb in range(num_batches):
        tBeg = time.time()
        span = range(batch_size*bb, min(rgb.shape[3],batch_size*(bb+1)))
        net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,span], (3,2,1,0))
        output = net.forward()
        prediction[:, span] = np.transpose(output[feature_layer])
        if bb==0:
            print('batch ', bb)
            print('cost time %.1f'%(time.time()-tBeg) )

    return prediction
    