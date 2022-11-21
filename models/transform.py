import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def contrast_gamma(image, gamma=5.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
     
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def contrast_linear(image, alpha = 0.5, beta = 20):
    
    '''
    out[pixel] = alpha * image[pixel] + beta
    alpha is for contrast, beta is for brightness
    '''
    output = np.zeros(image.shape, image.dtype)
    h, w, ch = image.shape
    for y in range(h):
        for x in range(w):
            for c in range(ch):
                output[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    return output
