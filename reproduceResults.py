import corrConv
import numpy as np
import cv2
import pyrMaskiir
from scipy import ndimage
import os
import time
import thresh_finder
import lPyr
import argparse
import vid_writer



#python paper.py Calc2FirstOrderDiffEqSepofVars.mp4 calc2 .. results tempSepOVars
parser=argparse.ArgumentParser(description='video name')
parser.add_argument('inName',help='name of input video')
parser.add_argument('tag',help='name, shortened if desired') #todo - remove this
parser.add_argument('inDir',help='path to input video') #todo - remove this
parser.add_argument('outDir',help='directory of output video')
parser.add_argument('tempDir',help='temp video directory') #todo - remove this
args=parser.parse_args()


corr_support=12
erode_support=3*corr_support/2
skip_frames=1 #todo - either set this higher or get rid of it
skip_thresh=15
confidence=.95
r=1 #todo - figure out what this is
inference_lev=0 #todo - remove this from pyrMaskiir
nlevs=0
rand_mult=10 


results_dir=args.outDir
in_name=args.inDir+'/'+args.inName+'.mp4'
start_name=args.tag #todo - remove this

out_name=results_dir+start_name+'-rand-'+str(rand_mult)+'-OR-fixed-correlate-'+str(corr_support)+'-mask-lev-'+str(inference_lev)+'-skip-'+str(skip_frames)+'-nlevs-'+str(nlevs)+'-thresh-'+str(thresh)+'-confidence-'+str(confidence)+'-erode-support-'+str(erode_support)+'-iir-'+str(r) #todo - vastly simplify this

cap = cv2.VideoCapture(in_name)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)
num_channels=3

out=vid_writer.vid_writer(out_name,'.mp4',fourcc,fps,(frame_width,frame_height),args.tempDir,args.tempDir+'/'+start_name+'.txt',400) #todo - maybe use opencv directly

tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret = True

f=open(out_name+".csv",mode='w') #todo - get rid of it



curr_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
last_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int64)
this_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int64)
old_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
for lev in range(inference_lev): #todo - don't do
    old_frame=cv2.pyrDown(old_frame)
now_frame=np.zeros(old_frame.shape,dtype=np.uint8)
next_frame=np.zeros(old_frame.shape,dtype=np.uint8)
mem_frame=np.zeros(old_frame.shape,dtype=np.uint8)
mask_frame=np.zeros(old_frame.shape,dtype=np.uint8)
corr_frame=np.zeros((old_frame.shape[0],old_frame.shape[1],num_channels),dtype=np.float)
bit_front=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.bool)
bit_back=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.bool)
bitmask=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.uint8)
out_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
medium_mask=np.zeros(now_frame.shape,dtype=np.uint8)
new_mask=medium_mask.copy()
way_old=medium_mask.copy()
t=1 

switch=True


#todo - get rid
def get_new_frame(frame,nlevs):
    band=frame
    for lev in range(nlevs):
        band=cv2.pyrDown(band)
    return band

#todo - get rid
def sobel_norm(frame):
    sobx=cv2.Sobel(frame,cv2.CV_64F,1,0)
    soby=cv2.Sobel(frame,cv2.CV_64F,0,1)
    sobr=sobx*sobx+soby*soby
    return np.sqrt(np.sum(sobr)/(frame.shape[0]*frame.shape[1]*3))

thresh=thresh_finder.find(in_name,skip_thresh) #todo - this should be done online

ret,frame=cap.read()
last_frame[:,:,:]=frame
pyr_blender=pyrMaskiir.pyrMaskiir(frame,True,nlevs,inference_lev,r) #remember to get rid of inferenece lev, and change the names and stuff
old_frame[:,:,:]=get_new_frame(frame,inference_lev)#we're getting rid of get_new_frame
t = t+1


sad=0
while(sad<thresh):
    cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
    ret,frame=cap.read()
    this_frame[:,:,:]=frame
    sad=np.sum(np.abs(this_frame-last_frame))
    t = t+1


last_frame[:,:,:]=frame
curr_frame[:,:,:]=frame
now_frame[:,:,:]=get_new_frame(frame,inference_lev) #todo - not doing this anymore
corr_frame[:,:,:]=corrConv.equiv_corr(now_frame,next_frame,corr_support) #todo - next frame is zeros?
bit_back[:,:]=np.product(corr_frame>confidence,2)
tle=0 #todo - remove this


randframe=np.uint8(np.random.rand(now_frame.shape[0],now_frame.shape[1],3)*rand_mult)


while(cap.isOpened() and ret):
    sad=0
    while(sad<thresh):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);#remove this line and nothing will change?
        ret,frame=cap.read()
        this_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
        sad=np.sum(np.abs(this_frame-last_frame))
        t = t+1
        if t*skip_frames>=tot_time:
            break

    last_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
    next_frame[:,:,:]=get_new_frame(frame,inference_lev)#get_new_frame does nothing, get rid of it
    corr_frame[:,:,:]=corrConv.equiv_corr(now_frame*(1.-1./rand_mult)+randframe,next_frame*(1.-1./rand_mult)+randframe,corr_support)
    bit_front[:,:]=np.product(corr_frame>confidence,2)
    bitmask[:,:]=np.uint8((bit_back | bit_front))*255
    bit_back[:,:]=bit_front
    bitmask[:,:]=cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    bitmask[:,:]=cv2.erode(bitmask,np.ones((erode_support,erode_support)))
    for color in range(num_channels):
        mask_frame[:,:,color]=bitmask

    
    out_frame[:,:,:]=pyr_blender.maskOut(curr_frame,mask_frame)
    mem_frame[:,:,:]=now_frame
    now_frame[:,:,:]=next_frame
    curr_frame[:,:,:]=frame

    if pyr_blender.is_full():
        break
    if t*skip_frames >= tot_time:
        break


t=1

cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
ret,frame=cap.read()
last_frame[:,:,:]=frame
old_frame[:,:,:]=get_new_frame(frame,inference_lev)
t = t+1


while(cap.isOpened() and ret):
    sad=0
    while(sad<thresh):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
        ret,frame=cap.read()
        this_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
        sad=np.sum(np.abs(this_frame-last_frame))
        t = t+1
        if t*skip_frames>=tot_time:
            break
    print "out"
    out2.write(frame)
    
    last_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
    next_frame[:,:,:]=get_new_frame(frame,inference_lev)
    corr_frame[:,:,:]=corrConv.equiv_corr(now_frame*(1.-1./rand_mult)+randframe,next_frame*(1.-1./rand_mult)+randframe,corr_support)
    bit_front[:,:]=np.product(corr_frame>confidence,2)
    bitmask[:,:]=np.uint8((bit_back | bit_front))*255
    bit_back[:,:]=bit_front
    bitmask[:,:]=cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    bitmask[:,:]=cv2.erode(bitmask,np.ones((erode_support,erode_support)))

    for color in range(num_channels):
        mask_frame[:,:,color]=bitmask

    
    out_frame[:,:,:]=pyr_blender.maskOut(curr_frame,mask_frame)
    cv2.imshow('out',out_frame)
    cv2.waitKey(22)
    mem_frame[:,:,:]=now_frame
    now_frame[:,:,:]=next_frame
    curr_frame[:,:,:]=frame
    
    f.write(str(skip_frames*t)+','+str(tle)+'\n') #todo - I don't think this is still needed
    out.write(out_frame)
    if t*skip_frames >= tot_time:
        break


out.finish()
f.close()
cap.release()
cv2.destroyAllWindows()
