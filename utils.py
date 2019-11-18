
from typing import Optional, Callable, Tuple, List, NoReturn
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np
import cv2 as cv
import PIL as pil

def binarise(src: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
        Take a grayscale image anf
    """
    dst = src.copy()
    
    # Assign correct max and min, according to the source image dtype.
    _floats    = [np.float, np.float16, np.float32, np.float64, np.float128]
    _iffloat   = partial(
        lambda im, f_val, i_val: f_val if im.dtype in _floats else i_val, 
        src
    )
    
    _max, _min = list(map(_iffloat, [1.0, 0.0], [255, 0]))
    
    if threshold:
        dst[ dst >= threshold ] = _max
        dst[ dst <  threshold ] = _min
    else:
        threshold = src.mean()  # Default value for binarization
        dst[ dst >= threshold ] = _max
        dst[ dst <  threshold ] = _min
    
    return dst
##

def side_by_side(
    image1: np.ndarray, 
    image2: np.ndarray, 
    title1: Optional[str] = None, 
    title2: Optional[str] = None,
    _figsize: Optional[Tuple[int]] = (15, 10),
    **kw
) -> NoReturn: 
    """
    """
    
    fig = plt.figure(figsize = _figsize)
    
    fig.add_subplot(2, 1, 1)
    plt.imshow(image1, cmap = 'gray')
    if title1:
        plt.title(title1, size = 18)
    
    fig.add_subplot(2, 1, 2)
    plt.imshow(image2, cmap = 'gray')
    if title2:
        plt.title(title2, size = 18)
##


