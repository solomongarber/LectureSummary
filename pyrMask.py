import numpy as np
import cv2
import lPyr
class pyrMask:
    def __init__(self,im,thresh,nlevs,mask_lev):
        if nlevs>0:
            self.pyr=lPyr.build_l_pyr_levs(im,nlevs);
        else:
            self.pyr=lPyr.build_l_pyr(im)
        self.thresh=thresh
        self.nlevs=nlevs
        self.mask_lev=mask_lev
        self.mask_pyr=[]
        for lev in self.pyr:
            self.mask_pyr.append(np.zeros(lev.shape))


    def maskOut(self,im,mask):
        if self.nlevs>0:
            new_pyr=lPyr.build_l_pyr_levs(im,self.nlevs);
            mask_pyr=lPyr.build_mask_pyr_levs(mask,self.nlevs,self.mask_lev)
        else:
            new_pyr=lPyr.build_l_pyr(im)
            mask_pyr=lPyr.build_mask_pyr(mask,self.mask_lev)
        for band_num in range(len(new_pyr)):
            old_band=self.pyr[band_num]
            new_band=new_pyr[band_num]
            new_band[:,:,:]=new_band
            mask_band=mask_pyr[band_num]
            if self.thresh:
                mask_band=cv2.threshold(mask_band,254,255,cv2.THRESH_BINARY)[1]
            new_mask=np.divide(np.array(mask_band,dtype=np.float32),255)
            old_mask=np.subtract(1,new_mask)
            #print np.min(n.add(old_mask,new_mask))
            #print np.max(np.add(old_mask,new_mask))
            if band_num<(len(new_pyr)-1):
                new_band=np.array(np.multiply(new_band,new_mask),dtype=np.int16)
                old_band=np.array(np.multiply(old_band,old_mask),dtype=np.int16)
            else:
                new_band=np.array(np.multiply(new_band,new_mask),dtype=np.uint8)
                old_band=np.array(np.multiply(old_band,old_mask),dtype=np.uint8)
            self.pyr[band_num]=old_band+new_band
            self.mask_pyr[band_num]=np.float64((self.mask_pyr[band_num]+new_mask)>0)
        return lPyr.recon_l_pyr(self.pyr)

    def get_top_level_energy(self):
        top_band=self.pyr[0]
        return np.sum(np.abs(top_band,dtype=np.int32))

    def is_full(self):
        return np.product(self.mask_pyr[0].shape)==np.sum(self.mask_pyr[0])
