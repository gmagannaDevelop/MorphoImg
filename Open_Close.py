#!/usr/bin/env python
# coding: utf-8

# In[194]:


from typing import Optional, Callable, Tuple, List, NoReturn
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np
import cv2 as cv
import PIL as pil

import importlib


# In[2]:


# User-defined functions, utils module found in the same directory as Erosion.ipynb
from utils import binarise, side_by_side, rescale_img, reverse


# In[195]:


# Importamos todas nuestras funciones:
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[3]:


def opening(src: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
        As defined in pages 644 and 645 :
            'The opening A by B is the erosion of A by B, followed by a dilation of the result by B'
        
        This function is should be equivalent to :
            cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    """
    return cv.dilate(cv.erode(src, kernel, iterations=iterations), kernel, iterations=iterations)
##

def closing(src: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
        As defined in pages 644 and 6 45 :
            'The closing of A by B is simply the dilation of A by B, followed by erosion of the result by B.'
        
        This function is should be equivalent to :
            cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    """
    return cv.erode(cv.dilate(src, kernel, iterations=iterations), kernel, iterations=iterations)
##


# In[4]:


x = img.imread('imagenes/Im1T4.png')


# In[12]:


plt.imshow(x, cmap='gray')


# In[13]:


x = reverse(x)


# In[14]:


plt.imshow(x, cmap='gray')


# In[15]:


binaria = binarise(x)
plt.imshow(binaria, cmap='gray')


# In[16]:


help(opening)


# # Opening

# In[17]:


kernel = np.ones((10, 10))
side_by_side(binaria, opening(binaria, kernel), title1='Original', title2=f'opening() with Kernel {kernel.shape}')


# In[18]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel), title1='Original', title2=f'cv.MORPH_OPEN with Kernel {kernel.shape}')


# In[19]:


kernel = np.ones((2, 50))
side_by_side(binaria, opening(binaria, kernel), title1='Original', title2=f'opening() with Kernel {kernel.shape}')


# In[20]:


kernel = np.ones((2, 50))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel), title1='Original', title2=f'cv.MORPH_OPEN with Kernel {kernel.shape}')


# In[21]:


kernel = np.ones((50, 2))
side_by_side(binaria, opening(binaria, kernel), title1='Original', title2=f'opening() with Kernel {kernel.shape}')


# In[22]:


kernel = np.ones((50, 2))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel), title1='Original', title2=f'cv.MORPH_OPEN with Kernel {kernel.shape}')


# As Gonzalez explained in the book, an opening is nothing but an erosion followed by a dilation.
# Our custom function ```opening(image, kernel)``` yields the same result as executing ```cv.morphologyEx(image, cv.MORPH_OPEN, kernel)```

# # Closing

# In[23]:


kernel = np.ones((10, 10))
side_by_side(binaria, closing(binaria, kernel), title1='Original', title2=f'closing() with Kernel {kernel.shape}')


# In[24]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel), title1='Original', title2=f'cv.MORPH_CLOSE with Kernel {kernel.shape}')


# In[25]:


kernel = np.ones((2, 50))
side_by_side(binaria, closing(binaria, kernel), title1='Original', title2=f'closing() with Kernel {kernel.shape}')


# In[26]:


kernel = np.ones((2, 50))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel), title1='Original', title2=f'cv.MORPH_CLOSE with Kernel {kernel.shape}')


# In[27]:


kernel = np.ones((50, 2))
side_by_side(binaria, closing(binaria, kernel), title1='Original', title2=f'closing() with Kernel {kernel.shape}')


# In[28]:


kernel = np.ones((50, 2))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel), title1='Original', title2=f'cv.MORPH_CLOSE with Kernel {kernel.shape}')


# As Gonzalez explained in the book, the closing operation is nothing but a dilation followed by an erosion.
# Our custom function ```closing(image, kernel)``` yields the same result as executing ```cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)```

# # Example found on page 643

# In[30]:


figura = cv.imread('imagenes/figura.png', 0) / 255.0
figura = reverse(figura)
figura.shape, figura.dtype


# In[31]:


plt.imshow(figura, cmap='gray')


# In[32]:


figura2 = binarise(figura)
plt.imshow(figura2, cmap='gray')


# In[35]:


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
side_by_side(figura2, cv.morphologyEx(figura2, cv.MORPH_OPEN ,kernel), title1='Original', title2=f'OPEN : Kernel {kernel.shape}')


# In[36]:


side_by_side(figura2, cv.morphologyEx(figura2, cv.MORPH_CLOSE ,kernel), title1='Original', title2=f'CLOSE : Kernel {kernel.shape}')


# In[37]:


plt.figure(figsize=(15,10))
plt.imshow(cv.imread('imagenes/opening.png'))
plt.title('Opening, acording to Gonzalez', size = 18)


# In[38]:


plt.figure(figsize=(15,10))
plt.imshow(cv.imread('imagenes/closing.png'))
plt.title('Closing, acording to Gonzalez', size = 18)


# 

# # Idempotence property
# 
# 1. $$ (A \circ B) \circ B = A \circ B $$
# 
# Through induction, one arrives to the conclusion that this operation can be repeated indefinitely and the result will always be the same as one opening.

# In[105]:


kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
kernel


# In[79]:


idemPot = figura2.copy()
for i in range(50):
    idemPot = cv.morphologyEx(idemPot, cv.MORPH_OPEN, kernel)
    side_by_side(figura2, idemPot, title1='Original', title2=f'OPEN : Kernel {kernel.shape}, iter = {i+1}')


# In[118]:


plt.close('all')


# In[108]:


kernel = cv.getStructuringElement(cv.MORPH_CROSS, (15, 15))
kernel


# In[110]:


idemPot = figura2.copy()
for i in range(50):
    idemPot = cv.morphologyEx(idemPot, cv.MORPH_OPEN, kernel)
    side_by_side(figura2, idemPot, title1='Original', title2=f'OPEN : Kernel {kernel.shape}, iter = {i+1}')


# In[117]:


plt.close('all')


# In[115]:


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
kernel


# In[116]:


idemPot = figura2.copy()
for i in range(50):
    idemPot = cv.morphologyEx(idemPot, cv.MORPH_OPEN, kernel)
    side_by_side(figura2, idemPot, title1='Original', title2=f'OPEN : Kernel {kernel.shape}, iter = {i+1}')


# In[109]:


plt.close('all')


# It seems that OpenCV's implementation of the ellyptical/circular structuring element is kind of poor, i.e. its lack of precision breaks the idempotence property of Opening. 
# Creating a better structuring element (i.e. having it to be symmetrical at least) will result in idempotence being respected.

# In[200]:


def structuring_circle(size: int, radius: int):
    ''' 
        size : size of original 3D numpy matrix A.
        radius : radius of circle inside A which will be filled with ones.
        
        Inspired from : 
            https://stackoverflow.com/questions/53326570/how-to-create-sphere-inside-a-ndarray-python
    '''

    assert size >= 2*radius, 'Circle overflows matrix surface !'

    A = np.zeros((size, size)) 
    AA = A.copy() 
    D = AA.copy()
    
    ''' (x0, y0) : coordinates of center of circle inside A. '''
    x0, y0 = int(np.floor(A.shape[0]/2)), int(np.floor(A.shape[1]/2))


    for x in range(x0-radius, x0+radius):
        for y in range(y0-radius, y0+radius):
            ''' deb: measures how far a coordinate in A is far from the center. 
                deb>=0: inside the sphere.
                deb<0: outside the sphere.'''   
            deb = radius - abs(x0-x) - abs(y0-y)
            D[x, y] = deb
            if (deb)>=0: AA[x,y] = 1
                
    return AA, D


# In[201]:


struc, dist = structuring_circle(size=10, radius=5)
side_by_side(cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), struc)


# In[197]:


plt.imshow(dist)


# In[198]:


dist


# In[ ]:




