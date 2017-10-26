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
for lev in range(inference_lev):
    old_frame=cv2.pyrDown(old_frame)
now_frame=np.zeros(old_frame.shape,dtype=np.uint8)
next_frame=np.zeros(old_frame.shape,dtype=np.uint8)
mem_frame=np.zeros(old_frame.shape,dtype=np.uint8)
mask_frame=np.zeros(old_frame.shape,dtype=np.uint8)
corr_frame=np.zeros((old_frame.shape[0],old_frame.shape[1],num_channels),dtype=np.float)
bit_front=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.bool)
bit_back=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.bool)
bitmask=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.uint8)
oldbit=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.uint8)
new_bit=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.uint8)
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
    #print t
    #cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*t);
    #ret, frame = cap.read()
    sad=0
    while(sad<thresh):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
        ret,frame=cap.read()
        #frame=cv2.GaussianBlur(frame,(5,5),1)
        #frame=cv2.add(frame,fsig)
        ##this_frame[:,:,:]=frame
        this_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
        sad=np.sum(np.abs(this_frame-last_frame))
        #print t
        #print sad
        t = t+1
        if t*skip_frames>=tot_time:
            break
    print "out"
    if pyr_blender.is_full():
        cv2.imshow('a',np.uint8(128*((pyr_blender.mask_pyr)[0])))
    else:
        cv2.imshow('a',np.uint8(255*((pyr_blender.mask_pyr)[0])))
    #if switch:
    #    if t>1999:
    #        cv2.imwrite('./correlate/final/last-frame.png',np.uint8(mem_frame))
    #        cv2.imwrite('./correlate/final/next-frame.png',np.uint8(cv2.pyrDown(frame)))
    #        cv2.imwrite('./correlate/final/now-frame.png',np.uint8(now_frame))
    #        cv2.imwrite('./correlate/final/out-frame.png',out_frame)
    #        cv2.imwrite('./correlate/final/masksks.png',oldbit)
                        
    ##last_frame[:,:,:]=frame
    last_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
    next_frame[:,:,:]=get_new_frame(frame,inference_lev)
    tic=time.time()
    ##corr_frame[:,:]=corrConv.fastCorr(now_frame,next_frame,corr_support)
    #randframe=np.uint8(np.random.rand(now_frame.shape[0],now_frame.shape[1],3)*rand_mult)
    corr_frame[:,:,:]=corrConv.equiv_corr(now_frame*(1.-1./rand_mult)+randframe,next_frame*(1.-1./rand_mult)+randframe,corr_support)
    print time.time()-tic
    bit_front[:,:]=np.product(corr_frame>confidence,2)
    oldbit[:,:]=new_bit
    bitmask[:,:]=np.uint8((bit_back | bit_front))*255
    new_bit[:,:]=bitmask
    if switch:
        if t>0:
            switch=False
            bm=lPyr.build_g_pyr(bitmask)
            curr_lev=0
            #cv2.imwrite('./correlate/final/last-frame.png',np.uint8(last_frame))
            #cv2.imwrite('./correlate/final/curr-frame.png',np.uint8(frame))
            for m in bm:
                curr_lev=curr_lev+1
                #cv2.imwrite('./mask200-actually-'+str(t)+'-lev-'+str(curr_lev)+'.png',bitmask)
    bit_back[:,:]=bit_front
    bitmask[:,:]=cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    bitmask[:,:]=cv2.erode(bitmask,np.ones((erode_support,erode_support)))
    #bitmask[:,:]=cv2.morphologyEx(cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support))),cv2.MORPH_OPEN,np.ones((morph_support,morph_support)))
    #bitmask[:,:]=
    #grayframe=cv2.GaussianBlur(np.float32(cv2.cvtColor(now_frame,cv2.COLOR_BGR2GRAY)),(3,3),1);
    #ex=cv2.blur(grayframe,(flat_support/2,flat_support/2))
    #exx=cv2.blur(grayframe*grayframe,(flat_support/2,flat_support/2))
    #flatmask=np.uint8(np.sqrt(exx-ex*ex)<variance_thresh)*255
    
    #incorrect=np.sqrt(cv2.blur(np.square(ex-grayframe),(corr_support/2,corr_support/2)))
    #flatmask=np.uint8(incorrect<variance_thresh)*255
    #flatmask[:,:]=cv2.erode(flatmask,np.ones((flat_support/2,flat_support/2)))
    #flatmask[:,:]=cv2.morphologyEx(flatmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    #flatmask[:,:]=cv2.dilate(flatmask,np.ones((2*flat_support/3,2*flat_support/3)))
    #bitmask[bitmask<flatmask]=255
    for color in range(num_channels):
        mask_frame[:,:,color]=bitmask
        #mask_frame[:,:,color]=flatmask

    cv2.imshow(out_name,(mask_frame*255)*now_frame+(1-mask_frame*255)*now_frame/3)
    cv2.waitKey(22)
    #old_mask=np.uint8(medium_mask.copy()>0)
    #medium_mask=np.uint8(new_mask.copy()>0)
    #new_mask=np.uint8(mask_frame.copy()>0)
    #real_mask=np.uint8(old_mask+medium_mask+new_mask>2)*255
    #real_mask[medium_mask>real_mask]=255
    out_frame[:,:,:]=pyr_blender.maskOut(curr_frame,mask_frame)
    #out_frame[:,:,:]=pyr_blender.maskOut(way_old,real_mask)
    mem_frame[:,:,:]=now_frame
    now_frame[:,:,:]=next_frame
    #cv2.imshow(out_name,(real_mask*255)*way_old+(1-real_mask*255)*way_old/3)
    #way_old[:,:,:]=curr_frame
    curr_frame[:,:,:]=frame
    
    #tle=pyr_blender.get_top_level_energy()
    tle=sobel_norm(out_frame)
    #if not(started):
    #    bit_frame[:,:]=bit_frame|(bitmask==255)
    #    if np.product(bit_frame):
    #        started=True
    #        print "started"
    #        f.write(str(skip_frames*t)+','+str(tle)+'\n')
    #        out.write(out_frame)
            #out2.write(np.uint8((mask_frame*255)*new_frame))
    #else:
    #if(t==200):
    #    cv2.imwrite('./correlate/bg200.png',out_frame)
    #    cv2.imwrite('./correlate/mask200.png',mask_frame)
    #f.write(str(skip_frames*t)+','+str(tle)+'\n')
    #out.write(out_frame)
    #out2.write((mask_frame*255)*new_frame)
    #t=t+1
    #old_frame[:,:,:]=new_frame
    #if t%10==0:
    #    cv2.destroyAllWindows()
    if pyr_blender.is_full():
        break
    if t*skip_frames >= tot_time:
        break














    



t=180

cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
ret,frame=cap.read()
last_frame[:,:,:]=frame
print ret
#pyr_blender=pyrMaskiir.pyrMaskiir(cv2.imread('/Users/solomongarber/Downloads/Ground_Truth/ground_truth_bundles/from_alex/approved/51 to 55 -t 16/bbgt1.png')[:356,:,:],True,nlevs,inference_lev,r)
old_frame[:,:,:]=get_new_frame(frame,inference_lev)
t = t+1


while(cap.isOpened() and ret):
    print t
    #cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*t);
    #ret, frame = cap.read()
    sad=0
    while(sad<thresh):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
        ret,frame=cap.read()
        #frame=cv2.GaussianBlur(frame,(5,5),1)
        #frame=cv2.add(frame,fsig)
        ##this_frame[:,:,:]=frame
        this_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
        sad=np.sum(np.abs(this_frame-last_frame))
        #print t
        #print sad
        t = t+1
        if t*skip_frames>=tot_time:
            break
    print "out"
    out2.write(frame)
    #if switch:
    #    if t>1999:
    #        cv2.imwrite('./correlate/final/last-frame.png',np.uint8(mem_frame))
    #        cv2.imwrite('./correlate/final/next-frame.png',np.uint8(cv2.pyrDown(frame)))
    #        cv2.imwrite('./correlate/final/now-frame.png',np.uint8(now_frame))
    #        cv2.imwrite('./correlate/final/out-frame.png',out_frame)
    #        cv2.imwrite('./correlate/final/masksks.png',oldbit)
                        
    ##last_frame[:,:,:]=frame
    last_frame[:,:,:]=cv2.GaussianBlur(frame,(11,11),3)
    next_frame[:,:,:]=get_new_frame(frame,inference_lev)
    tic=time.time()
    ##corr_frame[:,:]=corrConv.fastCorr(now_frame,next_frame,corr_support)
    #randframe=np.uint8(np.random.rand(now_frame.shape[0],now_frame.shape[1],3)*rand_mult)
    corr_frame[:,:,:]=corrConv.equiv_corr(now_frame*(1.-1./rand_mult)+randframe,next_frame*(1.-1./rand_mult)+randframe,corr_support)
    print time.time()-tic
    bit_front[:,:]=np.product(corr_frame>confidence,2)
    oldbit[:,:]=new_bit
    bitmask[:,:]=np.uint8((bit_back | bit_front))*255
    new_bit[:,:]=bitmask
    if switch:
        if t>0:
            switch=False
            bm=lPyr.build_g_pyr(bitmask)
            curr_lev=0
            #cv2.imwrite('./correlate/final/last-frame.png',np.uint8(last_frame))
            #cv2.imwrite('./correlate/final/curr-frame.png',np.uint8(frame))
            for m in bm:
                curr_lev=curr_lev+1
                #cv2.imwrite('./mask200-actually-'+str(t)+'-lev-'+str(curr_lev)+'.png',bitmask)
    bit_back[:,:]=bit_front
    bitmask[:,:]=cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    bitmask[:,:]=cv2.erode(bitmask,np.ones((erode_support,erode_support)))
    #bitmask[:,:]=cv2.morphologyEx(cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support))),cv2.MORPH_OPEN,np.ones((morph_support,morph_support)))
    #bitmask[:,:]=
    #grayframe=cv2.GaussianBlur(np.float32(cv2.cvtColor(now_frame,cv2.COLOR_BGR2GRAY)),(3,3),1);
    #ex=cv2.blur(grayframe,(flat_support/2,flat_support/2))
    #exx=cv2.blur(grayframe*grayframe,(flat_support/2,flat_support/2))
    #flatmask=np.uint8(np.sqrt(exx-ex*ex)<variance_thresh)*255
    
    #incorrect=np.sqrt(cv2.blur(np.square(ex-grayframe),(corr_support/2,corr_support/2)))
    #flatmask=np.uint8(incorrect<variance_thresh)*255
    #flatmask[:,:]=cv2.erode(flatmask,np.ones((flat_support/2,flat_support/2)))
    #flatmask[:,:]=cv2.morphologyEx(flatmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    #flatmask[:,:]=cv2.dilate(flatmask,np.ones((2*flat_support/3,2*flat_support/3)))
    #bitmask[bitmask<flatmask]=255
    for color in range(num_channels):
        mask_frame[:,:,color]=bitmask
        #mask_frame[:,:,color]=flatmask

    cv2.imshow(out_name,(mask_frame*255)*now_frame+(1-mask_frame*255)*now_frame/3)
    cv2.waitKey(22)
    #old_mask=np.uint8(medium_mask.copy()>0)
    #medium_mask=np.uint8(new_mask.copy()>0)
    #new_mask=np.uint8(mask_frame.copy()>0)
    #real_mask=np.uint8(old_mask+medium_mask+new_mask>2)*255
    #real_mask[medium_mask>real_mask]=255
    out_frame[:,:,:]=pyr_blender.maskOut(curr_frame,mask_frame)

    #out_frame[:,:,:]=pyr_blender.maskOut(way_old,real_mask)
    mem_frame[:,:,:]=now_frame
    now_frame[:,:,:]=next_frame
    #cv2.imshow(out_name,(real_mask*255)*way_old+(1-real_mask*255)*way_old/3)
    #way_old[:,:,:]=curr_frame
    curr_frame[:,:,:]=frame
    
    #tle=pyr_blender.get_top_level_energy()
    tle=sobel_norm(out_frame)
    #if not(started):
    #    bit_frame[:,:]=bit_frame|(bitmask==255)
    #    if np.product(bit_frame):
    #        started=True
    #        print "started"
    #        f.write(str(skip_frames*t)+','+str(tle)+'\n')
    #        out.write(out_frame)
            #out2.write(np.uint8((mask_frame*255)*new_frame))
    #else:
    #if(t==200):
    #    cv2.imwrite('./correlate/bg200.png',out_frame)
    #    cv2.imwrite('./correlate/mask200.png',mask_frame)
    f.write(str(skip_frames*t)+','+str(tle)+'\n')
    out.write(out_frame)
    #out2.write((mask_frame*255)*new_frame)

    #old_frame[:,:,:]=new_frame
    #if t%10==0:
    #    cv2.destroyAllWindows()

    if t*skip_frames >= tot_time:
        break


out.finish()
f.close()
cap.release()
cv2.destroyAllWindows()
