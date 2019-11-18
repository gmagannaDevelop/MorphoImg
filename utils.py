
from typing import Optional, Callable, Tuple, List, NoReturn
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np
import cv2 as cv
import PIL as pil

def binarise(src: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
        Take a grayscale image of any range : 
            [0.0, 1.0]  (float)
            [0, 255]    (int)

        And binarise it, specifying a threshold or using the 
        image intensity mean value as default.

        The binarisation occurs as follows

        pixels_intensity >= threshold --> _max
        pixels_intensity <  threshold --> _min

        '_min' and '_max' are inferred from the source image's dtype

        dtype ~ float :
            _min = 0.0
            _max = 1.0

        dtype ~ int :
            _min = 0
            _max = 255
        
        Arguments:
                  src : A grayscale image, of type numpy.ndarray
            threshold : An optional value to specify the 
                        'cutoff' intensity value.
                        Defaults to mean intensity value.

        Returns : 
                  dst : A binary image, of type numpy.ndarray
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

def rescale_img(src: np.ndarray) -> np.ndarray:
    """
    """
    return src / 255.0
##

def reverse(src: np.ndarray) -> np.ndarray:
    """
    """
    return 1.0 - src


def side_by_side(
    image1: np.ndarray, 
    image2: np.ndarray, 
    title1: Optional[str] = None, 
    title2: Optional[str] = None,
    _figsize: Optional[Tuple[int]] = (15, 10),
    **kw
) -> NoReturn: 
    """
        Show two matplotlib.pyplot.imshow images, side by side.
        
        Optional arguments : 
            Title for each subplot.
            Figsize, specified as a tuple.
    """
    
    fig = plt.figure(figsize = _figsize)
    
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1, cmap = 'gray')
    if title1:
        plt.title(title1, size = 18)
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2, cmap = 'gray')
    if title2:
        plt.title(title2, size = 18)
##


