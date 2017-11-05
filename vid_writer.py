import cv2
import numpy as np
import os

#eg out=vid_writer.vid_writer('rumple_sphere','.mp4',fourcc,60,(256,256),'rumple','./rumple/rumple_sphere.txt',360)

class vid_writer:

 def __init__(self,out_name,ext,fourcc,fps,vid_shp,out_dir,out_file,num_frames):
  self.num_frames=num_frames
  self.frame_num=0
  self.out_name=out_name
  self.ext=ext
  self.vid_num=0
  self.fourcc=fourcc
  self.fps=fps
  self.out_file=out_file
  self.out_dir=out_dir
  self.vid_shp=vid_shp
  self.out=cv2.VideoWriter(out_dir+'/'+out_name+str(self.vid_num)+ext,fourcc,fps,vid_shp,True)
  os.system('mkdir '+out_dir)
  self.f=open(out_file,'w')
  self.f.write('file '+out_name+str(self.vid_num)+ext)



 def get_name(self):
  return self.out_dir+'/'+self.out_name+str(self.vid_num)+self.ext

 def get_fname(self):
  return self.out_name+str(self.vid_num)+self.ext

 def write(self,im):
  self.frame_num+=1
  if self.frame_num%self.num_frames==0:
   self.vid_num+=1
   self.out.release()
   self.f.write('\nfile '+self.get_fname())
   self.out=cv2.VideoWriter(self.get_name(),self.fourcc,self.fps,self.vid_shp,True)
  self.out.write(im)


 def finish(self):
  self.out.release()
  self.f.close()
  os.system("ffmpeg -safe 0 -f concat -i "+self.out_file+" -c copy "+self.out_name+self.ext)

def mkspheres(sfn):
 fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
 out=vid_writer('sp','.mp4',fourcc,60,(300,300),'sphere','sphere/sps.txt',400)
 x=300
 y=150
 for r in np.arange(144,-1,-2):
  for theta in np.arange(90)*np.pi/45.:
    nx=np.int32(r*np.cos(theta))
    ny=np.int32(r*np.sin(theta))
    if not(nx==x and ny ==y):
     x=nx
     y=ny
     if x*x+y*y<150*150:
      outfrm=light(sfn,150-y,150+x)
      #vec=sfn[y,x,:]
      #tmp=np.sum(sfn[:,:]*vec,2)
      #tmp[tmp<0]=0-tmp[tmp<0]
      #outfrm[:,:]=np.uint8(255*vec)
      #outfrm[:,:,:]=np.uint8(outfrm*np.dstack((tmp,tmp,tmp)))
      out.write(outfrm)
 out.finish()

def light(sfn,y,x):
 vec=sfn[y,x,:]
 tmp=np.sum(sfn[:,:]*vec,2)
 tmp[tmp<0]=0-tmp[tmp<0]
 #tmp[tmp<0]=0
 outfrm=np.zeros(sfn.shape,dtype=np.uint8)
 outfrm[:,:]=np.uint8(255*np.abs(vec))
 outfrm[:,:,:]=np.uint8(outfrm*np.dstack((tmp,tmp,tmp)))
 return outfrm
