import numpy as np
import cv2

def find(in_vid,skip):
    cap=cv2.VideoCapture(in_vid)
    ret,frame=cap.read()
    tot_time=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    old=np.zeros(frame.shape,dtype=np.int64)
    old[:,:,:]=frame
    new=np.zeros(frame.shape,dtype=np.int64)
    acc=0
    while(t*skip<tot_time):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip)
        ret,frame=cap.read()
        new[:,:,:]=frame
        acc=acc+np.sum(np.abs(old-new))
        t=t+1
        old[:,:,:]=frame
        if(t%50)==0:
            print t
    acc=acc/t
    return acc
