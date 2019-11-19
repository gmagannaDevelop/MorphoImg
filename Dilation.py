#!/usr/bin/env python
# coding: utf-8

# In[31]:


from typing import Optional, Callable, Tuple, List, NoReturn
from functools import partial, reduce

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


help(cv.dilate)


# In[9]:


kernel = np.ones((10, 10))
side_by_side(binaria, cv.dilate(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[10]:


kernel = np.ones((2, 50))
side_by_side(binaria, cv.dilate(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[11]:


kernel = np.ones((50, 2))
side_by_side(binaria, cv.dilate(binaria, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# # Example found on page 643

# In[12]:


text = cv.imread('imagenes/text.png', 0)
text.shape


# In[13]:


plt.imshow(text, cmap='gray')


# In[14]:


text2 = binarise(text, threshold=115)
plt.imshow(text2, cmap='gray')


# In[15]:


kernel = np.ones((1, 1))
side_by_side(text2, cv.dilate(text2, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[16]:


kernel = np.ones((3, 3))
side_by_side(text2[400:, 400:], cv.dilate(text2[400:, 400:], kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[17]:


kernel = np.ones((15, 15))
side_by_side(text2, cv.dilate(text2, kernel), title1='Original', title2=f'Kernel {kernel.shape}')


# In[ ]:





# In[43]:


A = np.zeros((11, 11))
A[5, 5] = 1
plt.grid()
plt.imshow(A, cmap='gray')


# In[42]:


B = [A, np.ones((3, 1)), np.ones((1, 3))]
dilate_decomposed = reduce(lambda x, y: cv.dilate(x, y), B)
plt.grid()
plt.imshow(dilate_decomposed, cmap='gray')


# In[49]:


y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0])
y.shape = 3, 3
y = np.uint8(y)
y


# In[52]:


side_by_side(A, cv.dilate(A, y))


# In[40]:


np.ones((3, 1))


# In[ ]:


reverse

