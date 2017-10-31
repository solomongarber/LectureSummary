import numpy as np


    

def corr(im1,im2,support):
    width=2*support+1
    shp=(width,width)
    x=np.float64(im1);y=np.float64(im2);
    ex=cv2.blur(x,shp)
    ey=cv2.blur(y,shp)
    exy=cv2.blur(x*y,shp)
    exx=cv2.blur(x*x,shp)
    eyy=cv2.blur(y*y,shp)
    return ((exy-ex*ey)/(np.sqrt(exx-ex*ex)*np.sqrt(eyy-ey*ey)))
    
