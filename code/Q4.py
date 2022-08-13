#!/usr/bin/env python
# coding: utf-8

# In[5]:


from scipy.signal import convolve2d, correlate2d
import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List
from scipy import fftpack
from scipy import ndimage
from skimage.transform import rescale
import time
import seaborn as sn
from matplotlib.colors import LogNorm, Normalize


# In[6]:


img = cv2.imread('RISDance.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) ;


# In[7]:


imgsize = (img.shape[0]*img.shape[1])/(10**6)
sizes=np.array([0.25 ,2, 4, 8])
ratio=sizes/imgsize


# In[8]:


img0= rescale(img, ratio[0], mode='reflect', multichannel=True)
img1= rescale(img, ratio[1], mode='reflect', multichannel=True)
img2= rescale(img, ratio[2], mode='reflect', multichannel=True)
img3= rescale(img, ratio[3], mode='reflect', multichannel=True)

def create_mean_filter(ksize):
    assert ksize % 2 != 0
    mean_filter = np.ones((ksize, ksize, 3), dtype=np.uint8) * (1/(ksize**2))
    return mean_filter

kernel0 = create_mean_filter(3)
kernel1 = create_mean_filter(7)
kernel2 = create_mean_filter(11)
kernel3 = create_mean_filter(15)


# In[9]:


imglist= [img0, img1, img2, img3]
kernellist=[kernel0, kernel1, kernel2, kernel3 ]


# In[14]:


exec_time = np.ndarray((4,4))
for i in range(4):
    for j in range (4):
        imgtmp,kerneltmp=imglist[j],kernellist[i]
        start_time = time.time()
        output = ndimage.convolve(imgtmp,kerneltmp, mode ='constant')
        endtime = time.time() - start_time
        exec_time[i,j]=endtime


# In[23]:


plt.figure(figsize=(9,7));
hm = sn.heatmap(data=exec_time, annot=True,norm=LogNorm())
hm.invert_yaxis()
hm.set_xticklabels(['3x3','7x7','11x11','15x15'])
hm.set_xlabel('filter size')
hm.set_yticklabels(sizes)
hm.set_ylabel('image size (MPix)')
plt.show();

