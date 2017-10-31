import numpy as np
import cv2

def build_l_pyr(im):
    pyr=[]
    while(not(min(im.shape[0],im.shape[1])<20)):
        temp_im=im
        im=cv2.pyrDown(im)
        temp_band=cv2.pyrUp(im)[:temp_im.shape[0],:temp_im.shape[1],:]
        t_band=cv2.subtract(np.array(temp_im,dtype=np.int16),np.array(temp_band,dtype=np.int16))
        pyr.append(t_band)
    pyr.append(im)
    return pyr

def build_l_pyr_levs(im,nlevs):
    pyr=[]
    for lev in range(nlevs):
        temp_im=im
        im=cv2.pyrDown(im)
        temp_band=cv2.pyrUp(im)[:temp_im.shape[0],:temp_im.shape[1],:]
        t_band=cv2.subtract(np.array(temp_im,dtype=np.int16),np.array(temp_band,dtype=np.int16))
        pyr.append(t_band)
    pyr.append(im)
    return pyr

def recon_l_pyr(pyr):
    nlevs=len(pyr)
    lowpass=np.array(pyr[nlevs-1],dtype=np.uint8)
    for i in range(nlevs-2,-1,-1):
        band=pyr[i]
        lowpass=cv2.pyrUp(lowpass)[:band.shape[0],:band.shape[1],:]
        highpass=cv2.add(np.array(lowpass,dtype=np.int16),band)
        highpass=cv2.min(highpass,np.array([255,255,255]))
        highpass=cv2.max(highpass,np.array([0,0,0]))
        lowpass=np.array(highpass,dtype=np.uint8)
    return lowpass

def build_g_pyr(im):
    pyr=[]
    while(not(min(im.shape[0],im.shape[1])<20)):
        temp_im=im
        im=cv2.pyrDown(im)
        pyr.append(temp_im)
    pyr.append(im)
    return pyr

def build_g_pyr_levs(im,nlevs):
    pyr=[]
    for lev in range(nlevs):
        temp_im=im
        im=cv2.pyrDown(im)
        pyr.append(temp_im)
    pyr.append(im)
    return pyr

def build_mask_pyr(im,up_levs):
    pyr=[]
    temp_im=im
    for lev in range(up_levs):
        temp_im=cv2.pyrUp(temp_im)
        pyr.append(temp_im)
    pyr.reverse()
    while(not(min(im.shape[0],im.shape[1])<20)):
        temp_im=im
        im=cv2.pyrDown(im)
        pyr.append(temp_im)
    pyr.append(im)
    return pyr

def build_mask_pyr_levs(im,nlevs,up_levs):
    pyr=[]
    temp_im=im
    for lev in range(up_levs):
        temp_im=cv2.pyrUp(temp_im)
        pyr.append(temp_im)
    pyr.reverse()
    for lev in range(nlevs-up_levs):
        temp_im=im
        im=cv2.pyrDown(im)
        pyr.append(temp_im)
    pyr.append(im)
    return pyr

def build_l_pyr_vec(im):
    square_pyr=build_l_pyr(im)
    nlevs=len(square_pyr)
    pyr=np.array([],dtype=np.int16)
    pind=np.array([[]],dtype=np.int16)
    for lev in range(nlevs):
        band=np.array(square_pyr[lev],dtype=np.int16)
        numel=np.size(band[:,:,0])
        first_ind=band.shape[0]
        last_ind=band.shape[1]
        band=np.reshape(band,(numel,3))
        pyr=np.concatenate(pyr,band)
        pind=np.concatenate(pind,[[first_ind,last_ind,3]])
    return (pyr,pind)

def recon_l_pyr_vec(pyr,pind):
    square_pyr=[]
    nlevs=pind.shape[0]
    past=0
    for lev in range(nlevs):
        inds=pind[lev]
        numel=inds[0]*inds[1]
        band=np.reshape(pyr[past:numel,:],inds)
        pyr.append(band)
        past=past+numel
    return recon_l_pyr(pyr)
