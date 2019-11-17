#!/usr/bin/env python
# coding: utf-8

# In[48]:


from typing import Optional, Callable, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np
import cv2 as cv
import PIL as pil


# In[49]:


x = img.imread('imagenes/Im1T4.png')


# In[50]:


plt.imshow(x, cmap='gray')


# In[51]:


x = 1 - x


# In[52]:


plt.imshow(x, cmap='gray')


# In[47]:


def umbraliza(src: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    """
    dst = src.copy()
    
    # Assign correct max and min, according to the source image dtype.
    _floats    = [np.float, np.float16, np.float32, np.float64, np.float128]
    _iffloat   = lambda im, f_val, i_val: f_val if im.dtype in _floats else i_val
    _max, _min = _iffloat(src, 1.0, 255), _iffloat(src, 0.0, 0)
    
    if threshold:
        dst[ dst >= threshold ] = _max
        dst[ dst <  threshold ] = _min
    else:
        threshold = src.mean()
        dst[ dst >= threshold ] = _max
        dst[ dst <  threshold ] = _min
    
    return dst
    


# In[57]:


plt.imshow(umbraliza(x), cmap='gray')


# In[17]:


binaria[ binaria >= x.mean()] = 1.0
x[ x < x.mean()]  = 0.0


# In[18]:


plt.imshow(x, cmap='gray')


# In[19]:


help(cv.erode)


# In[22]:


kernel = np.ones(1)
kernel


# In[21]:


plt.imshow


# In[32]:


lol = x.dtype


# In[36]:


x.dtype


# In[39]:


np.float


# In[40]:


x.dtype in [np.float, np.float32]


# In[43]:


e = 


# In[44]:





# In[ ]:




