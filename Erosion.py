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


# In[7]:


binaria = binarise(x)
plt.imshow(binaria, cmap='gray')


# In[8]:


help(cv.erode)


# In[9]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.erode(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[10]:


kernel = np.ones((2, 30))
side_by_side(binaria, cv.erode(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[11]:


kernel = np.ones((70, 2))
side_by_side(binaria, cv.erode(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# # Example found on page 641

# In[20]:


wbm = cv.imread('imagenes/wire_bond_mask.png', 0)
wbm.shape


# In[21]:


plt.imshow(wbm, cmap='gray')


# In[22]:


wbm = binarise(wbm)
plt.imshow(wbm, cmap='gray')


# In[24]:


kernel = np.ones((11, 11))
side_by_side(wbm, cv.erode(wbm, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[25]:


kernel = np.ones((15, 15))
side_by_side(wbm, cv.erode(wbm, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[26]:


kernel = np.ones((45, 45))
side_by_side(wbm, cv.erode(wbm, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[ ]:




