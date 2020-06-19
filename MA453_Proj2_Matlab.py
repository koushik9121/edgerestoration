# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:13:54 2020

@author: Charan Charupalli
"""

from skimage.io import imread
import scipy
from scipy.integrate import quad
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

im1 = imread("twoObj.bmp")
im1 = rgb2gray(im1)
#im1 = data.camera()
img = np.array(im1, dtype='float32')
# parameters
timestep = 0.01 #dt
iters = 80
plt.imshow(img,cmap='gray')
plt.show()
lam = 3.5
sigma = 3 
sigma1 = 1        # scale parameter in Gaussian kernel
#img_smooth = img
#img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution.
def g(f,lam):
    return np.exp(-np.power(f/lam,5))
def lin_dif(im, D, timestep):
    img = im
    [uy,ux] = np.gradient(img)
    img = NeumannBoundCond(img)
    img = img + timestep * (D*div(ux,uy))
    return img

def div(nx, ny):
    [junk, nxx] = np.gradient(nx)
    [nyy, junk] = np.gradient(ny)
    return nxx + nyy

def NeumannBoundCond(f):
    [ny, nx] = f.shape
    g = f.copy()

    g[0, 0] = g[2, 2]
    g[0, nx-1] = g[2, nx-3]
    g[ny-1, 0] = g[ny-3, 2]
    g[ny-1, nx-1] = g[ny-3, nx-3]

    g[0, 1:-1] = g[2, 1:-1]
    g[ny-1, 1:-1] = g[ny-3, 1:-1]

    g[1:-1, 0] = g[1:-1, 2]
    g[1:-1, nx-1] = g[1:-1, nx-3]

    return g



I = np.identity(img.shape[0])
D_iso_lin = I

def D_iso_non_lin(img):
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.sqrt(np.square(Ix)+np.square(Iy))
    return g(f,lam)


def D_an_non_lin_tensor(img):
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)
    [Iy, Ix] = np.gradient(img_smooth)
    product = ((np.square(Ix)+np.square(Iy))/3.0)-2*(np.multiply(Ix,Iy)/3.0)
    product = np.nan_to_num(np.sqrt(product))
    return g(product,lam)
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
for n in range(iters):   
    img3 = lin_dif(img3,D_an_non_lin_tensor(img3),timestep)
    img2 = lin_dif(img2,D_iso_non_lin(img2),timestep)
    img1 = lin_dif(img1, I, timestep)
        
print(img.shape)    
print(np.array_equal(img2,img3))
print("linear iso")
plt.imshow(img1,cmap='gray')
plt.show()
print("non-linear iso")
plt.imshow(img2,cmap='gray')
plt.show()
print("non-linear aniso tensor")
plt.imshow(img3,cmap='gray')
