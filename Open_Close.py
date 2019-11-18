#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from utils import binarise, side_by_side, rescale_img, reverse, structuring_circle


# In[3]:


# Importamos todas nuestras funciones:
import mfilt_funcs as mine
importlib.reload(mine)
from mfilt_funcs import *


# In[4]:


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


# In[5]:


x = img.imread('imagenes/Im1T4.png')


# In[6]:


plt.imshow(x, cmap='gray')


# In[7]:


x = reverse(x)


# In[8]:


plt.imshow(x, cmap='gray')


# In[9]:


binaria = binarise(x)
plt.imshow(binaria, cmap='gray')


# In[10]:


help(opening)


# # Opening

# In[11]:


kernel = np.ones((10, 10))
side_by_side(binaria, opening(binaria, kernel), title1='Original', title2=f'opening() with Kernel {kernel.shape}')


# In[12]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel), title1='Original', title2=f'cv.MORPH_OPEN with Kernel {kernel.shape}')


# In[13]:


kernel = np.ones((2, 50))
side_by_side(binaria, opening(binaria, kernel), title1='Original', title2=f'opening() with Kernel {kernel.shape}')


# In[14]:


kernel = np.ones((2, 50))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel), title1='Original', title2=f'cv.MORPH_OPEN with Kernel {kernel.shape}')


# In[15]:


kernel = np.ones((50, 2))
side_by_side(binaria, opening(binaria, kernel), title1='Original', title2=f'opening() with Kernel {kernel.shape}')


# In[16]:


kernel = np.ones((50, 2))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel), title1='Original', title2=f'cv.MORPH_OPEN with Kernel {kernel.shape}')


# As Gonzalez explained in the book, an opening is nothing but an erosion followed by a dilation.
# Our custom function ```opening(image, kernel)``` yields the same result as executing ```cv.morphologyEx(image, cv.MORPH_OPEN, kernel)```

# # Closing

# In[17]:


kernel = np.ones((10, 10))
side_by_side(binaria, closing(binaria, kernel), title1='Original', title2=f'closing() with Kernel {kernel.shape}')


# In[18]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel), title1='Original', title2=f'cv.MORPH_CLOSE with Kernel {kernel.shape}')


# In[19]:


kernel = np.ones((2, 50))
side_by_side(binaria, closing(binaria, kernel), title1='Original', title2=f'closing() with Kernel {kernel.shape}')


# In[20]:


kernel = np.ones((2, 50))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel), title1='Original', title2=f'cv.MORPH_CLOSE with Kernel {kernel.shape}')


# In[21]:


kernel = np.ones((50, 2))
side_by_side(binaria, closing(binaria, kernel), title1='Original', title2=f'closing() with Kernel {kernel.shape}')


# In[22]:


kernel = np.ones((50, 2))
side_by_side(binaria, cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel), title1='Original', title2=f'cv.MORPH_CLOSE with Kernel {kernel.shape}')


# As Gonzalez explained in the book, the closing operation is nothing but a dilation followed by an erosion.
# Our custom function ```closing(image, kernel)``` yields the same result as executing ```cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)```

# # Example found on page 643

# In[23]:


figura = cv.imread('imagenes/figura.png', 0) / 255.0
figura = reverse(figura)
figura.shape, figura.dtype


# In[24]:


plt.imshow(figura, cmap='gray')


# In[25]:


figura2 = binarise(figura)
plt.imshow(figura2, cmap='gray')


# In[26]:


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
side_by_side(figura2, cv.morphologyEx(figura2, cv.MORPH_OPEN ,kernel), title1='Original', title2=f'OPEN : Kernel {kernel.shape}')


# In[27]:


side_by_side(figura2, cv.morphologyEx(figura2, cv.MORPH_CLOSE ,kernel), title1='Original', title2=f'CLOSE : Kernel {kernel.shape}')


# In[28]:


plt.figure(figsize=(15,10))
plt.imshow(cv.imread('imagenes/opening.png'))
plt.title('Opening, acording to Gonzalez', size = 18)


# In[29]:


plt.figure(figsize=(15,10))
plt.imshow(cv.imread('imagenes/closing.png'))
plt.title('Closing, acording to Gonzalez', size = 18)


# 

# # Idempotence property
# 
# 1. $$ (A \circ B) \circ B = A \circ B $$
# 
# Through induction, one arrives to the conclusion that this operation can be repeated indefinitely and the result will always be the same as one opening.

# In[30]:


kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
kernel


# In[31]:


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


# In[32]:


plt.close('all')


# In[156]:


#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
kernel = structuring_circle(radius=12)
#kernel


# In[157]:


#side_by_side(figura2, opening(figura2, kernel), title1='Original', title2=f'OPEN : Kernel {kernel.shape}, iter = {i+1}')


# In[158]:


idemPot = figura2.copy()
for i in range(50):
    idemPot = cv.morphologyEx(idemPot, cv.MORPH_OPEN, kernel)
    side_by_side(figura2, idemPot, title1='Original', title2=f'OPEN : Kernel {kernel.shape}, iter = {i+1}')


# In[149]:


plt.close('all')


# It seems that OpenCV's implementation of the ellyptical/circular structuring element is kind of poor, i.e. its lack of precision breaks the idempotence property of Opening. 
# Creating a better structuring element (i.e. having it to be symmetrical at least) will result in idempotence being respected.

# In[162]:


def structuring_circle(radius: int, size: Optional[int] = None):
    ''' 
        size : size of original 3D numpy matrix A.
        radius : radius of circle inside A which will be filled with ones.
        
        Inspired from : 
            https://stackoverflow.com/questions/53326570/how-to-create-sphere-inside-a-ndarray-python
    '''
    if size:
        assert size >= 2*radius, 'Circle overflows matrix surface !'
        assert size % 2 == 0, 'Size must be even !'
    else:
        size = 2*radius
        
    A = np.zeros((size+1, size+1))
    AA = A.copy() 
    D = AA.copy()
    
    ''' (x0, y0) : coordinates of center of circle inside A. '''
    x0, y0 = int(np.floor(A.shape[0]/2)), int(np.floor(A.shape[1]/2))


    for x in range(x0-radius, x0+radius+1):
        for y in range(y0-radius, y0+radius+1):
            ''' deb: measures how far a coordinate in A is far from the center. 
                deb>=0: inside the sphere.
                deb<0: outside the sphere.'''   
            deb = radius - abs(x0-x) - abs(y0-y) 
            D[x, y] = deb
            if (deb)>=0: AA[x,y] = 1
    
    AA = np.uint8(AA)
    
    return AA


# In[166]:


struc = structuring_circle(radius=4)
side_by_side(cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), struc)


# In[95]:


struc[0, :]


# In[96]:


side_by_side(struc, np.rot90(struc))


# In[97]:


plt.imshow(dist)


# In[39]:


dist


# In[44]:


_x = (-4., -3., -2., -1.,  0.,  1.,  0., -1., -2., -3.)


# In[43]:


len((-3., -2., -1.,  0.,  1.,  2.,  1.,  0., -1., -2.))


# In[46]:


_x[int(np.floor(10/2))]


# In[68]:


_x = np.array(range(10))
_x


# In[116]:


struc.dtype


# In[122]:


cv.getStructuringElement(cv.MORPH_RECT, (4,4))


# In[ ]:


kernel.dtype = np.int8


# In[134]:


np.int(3.45)


# In[ ]:




