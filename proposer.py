import numpy as np
import cv2
import corrConv

import argparse
import os

#python batch_paper.py vids.txt tempTrash finalResults
thresh1 = 3000
thresh2 = 1
thresh3 = 2
def sa_mask(frm):
    sx=cv2.Sobel(frm,cv2.CV_64F,0,1)
    sy=cv2.Sobel(frm,cv2.CV_64F,1,0)
    sa=np.mean(np.sqrt(sx*sx+sy*sy),2)
    return cv2.erode(np.float64(cv2.dilate(np.float64(sa>30),np.ones((12,12)))),np.ones((3,3)))

def energy(frm):
    sx=cv2.Sobel(frm,cv2.CV_64F,0,1)
    sy=cv2.Sobel(frm,cv2.CV_64F,1,0)
    return np.mean(np.sqrt(sx*sx+sy*sy),2)
    
def sigmask(frm):
    ex=cv2.blur(np.float64(frm),(12,12))
    exx=cv2.blur(np.square(np.float64(frm)),(12,12))
    return np.float64(np.mean(exx-ex*ex,2)>10)


parser=argparse.ArgumentParser(description='video name')
parser.add_argument('inFile',help='video name')
parser.add_argument('outDir',help='directory of output frames')

args=parser.parse_args()
os.system("mkdir "+args.outDir)
cap=cv2.VideoCapture(args.inFile)
ret,frame=cap.read()
picnum=0

outframes=[frame]
#outmasks=[np.ones((frame.shape[0],frame.shape[1]))]
outmasks=[sa_mask(frame)]
old_mask=sa_mask(frame)
ret,frame2=cap.read()
t=0

def ate(im):
    oim=np.float64(im)-np.min(im)
    return np.uint8(255*oim/np.max(oim))

deleting = False
cumsum=0
outs=[]
while ret:
    mask=sa_mask(frame2)
    corr_mask=np.mean(corrConv.equiv_corr(frame,frame2,12),2)>.9
    #corr_mask=cv2.dilate(np.uint8(corr_mask),np.ones((3,3)))
    cv2.imshow('cm',ate(corr_mask))
    new_info=mask*(1-np.float64(corr_mask))
    old_kept=old_mask*corr_mask*mask
    old_lost=cv2.erode(old_mask,np.ones((6,6)))*(1-corr_mask)
    
    if np.sum(old_lost)>thresh1:
        if np.mean(energy(frame)[old_lost>0])>thresh2 and deleting<1:
            #cv2.imwrite(args.outDir+'/'+str(picnum)+'.png',frame)
            outs.append(frame)
            picnum = picnum+1
            print picnum
            deleting=True
            print "write frame"
            print "entering deleting mode"
            print True < 1
        old_mask[:,:]=np.float64((old_kept+new_info)>0)
        cumsum=0
        
    else:
        cumsum=cumsum+np.mean(energy(frame2)*new_info)
        if cumsum>thresh3:
            if deleting:
                print "done deleting"
            deleting=False
            #if np.int64(cumsum)%10==0:
            #    print "not deleting "+str(cumsum)
                
        old_mask[:,:]=np.float64((old_mask+new_info)>0)
        #old_mask[:,:]=mask
        cv2.imshow('om',ate(np.dstack((old_mask,old_mask,old_mask))*frame2))
        cv2.waitKey(22)
    frame[:,:,:]=frame2[:,:,:]
    ret,frame2=cap.read()
#cv2.imwrite(args.outDir+'/'+str(picnum)+'.png',frame)
outs.append(frame)

def isin(im,outs):
    corr_frame=np.zeros((im.shape[0],im.shape[1]))
    sx=cv2.Sobel(im,cv2.CV_64F,0,1,ksize=3)
    sy=cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
    edge_mask=np.float64(cv2.dilate(np.mean(np.sqrt(sx*sx+sy*sy),2),np.ones((5,5)))>20)
    for out in outs:
        curr_frame=np.float64(corrConv.equiv_corr(np.mean(im,2),np.mean(out,2),12)<.8)
        edge_mask*=curr_frame
        if np.sum(cv2.erode(edge_mask,np.ones((2,2))))<0.01*np.product(edge_mask.shape):
            return True
    return False

def remove_extras(outs):
    temp_outs=[np.copy(slide) for slide in outs]
    real_outs=[]
    i=0
    while(i<len(temp_outs)):
        im=temp_outs[i]
        fake_outs=temp_outs[:i]+temp_outs[i+1:]
        if isin(im,fake_outs):
            temp_outs=fake_outs
            print "deleting "+str(i)
        else:
            print "saving "+str(i)
            i+=1
            real_outs.append(im)
    return real_outs

outs=remove_extras(outs)
for (i,slide) in enumerate(outs):
    cv2.imwrite(args.outDir+'/'+str(i)+'.png',slide)
print "done"
"""
while ret:
  ret,frame2=cap.read()
  sx=cv2.Sobel(frame2,cv2.CV_64F,1,0)
  sy=cv2.Sobel(frame2,cv2.CV_64F,0,1)
  sa=np.mean(np.sqrt(sx*sx+sy*sy),2)

  for [i,outframe] in enumerate(outframes):
      msk=np.float64(corrConv.equiv_corr(outframe[:,:,1],frame2[:,:,1],12)>.9)
      badmatch=msk*outmasks[i]
      for j in range(i,len(outmasks)):
          outmasks[j][badmatch>0]=0
      unmatch = (1-msk)*(1-outmasks[i])
  for [i,outframe] in enumerate(outframes):
      msk=np.float64(corrConv.equiv_corr(outframe[:,:,1],frame2[:,:,1],12)>.9)
      ununmatch = (1-msk)*(1-outmasks[i])
      unmatch=1-msk
      

  
  #cv2.imshow('msk',ate(np.float64(np.dstack((msk,msk,msk))<0.9)*frame))
  #cv2.waitKey(33)
  sx=cv2.Sobel(frame,cv2.CV_64F,1,0)
  sy=cv2.Sobel(frame,cv2.CV_64F,0,1)
  sa=np.mean(np.sqrt(sx*sx+sy*sy)*np.float64(np.dstack((msk,msk,msk))<0.9))
  if sa >2:
      cv2.imwrite(args.outDir+'/'+str(picnum)+'.png',frame)
      frame[:,:,:]=frame2[:,:,:]
      picnum=picnum+1"""
