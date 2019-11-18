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


# In[2]:


# User-defined functions, utils module found in the same directory as Erosion.ipynb
from utils import binarise, side_by_side


# In[3]:


x = img.imread('imagenes/Im1T4.png')


# In[4]:


plt.imshow(x, cmap='gray')


# In[5]:


x = 1 - x


# In[6]:


plt.imshow(x, cmap='gray')


# In[9]:


binaria = binarise(x)
plt.imshow(binaria, cmap='gray')


# In[10]:


help(cv.erode)


# In[11]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.erode(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[12]:


kernel = np.ones((2, 30))
side_by_side(binaria, cv.erode(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[13]:


kernel = np.ones((70, 2))
side_by_side(binaria, cv.erode(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')

