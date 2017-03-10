# Temporary file to test using proximal operators in the NMF deblender
from __future__ import print_function, division
from collections import OrderedDict
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig()
logger = logging.getLogger("lsst.meas.deblender.display")

def plotSeds(seds):
    """Plot the SEDs for each source
    """
    for col in range(seds.shape[1]):
        sed = seds[:, col]
        band = range(len(sed))
        lbl = "Obj {0}".format(col)
        plt.plot(band, sed, '.-', label=lbl)
    plt.xlabel("Filter Number")
    plt.ylabel("Flux")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=seds.shape[1])
    plt.show()

def plotIntensities(seds, intensities, shape, fidx=0,
                    vmin=None, vmax=None, useMask=False):
    """Plot the template image for each source
    
    Multiply each row in ``intensities`` by the SED for filter ``fidx`` and
    plot the result.
    """
    from .proximal import reconstructTemplate
    for k in range(len(intensities)):
        template = reconstructTemplate(seds, intensities, fidx, k, shape)
        # Optionally Mask zero pixels (gives a better idea of the footprint)
        if useMask:
            template = np.ma.array(template, mask=template==0)
        plt.title("Object {0}".format(k))
        plt.imshow(template, interpolation='none', cmap='inferno', vmin=vmin, vmax=vmax)
        plt.show()

def imagesToRgb(images=None, calexps=None, filterIndices=None, xRange=None, yRange=None,
                contrast=1, adjustZero=True):
    """Convert a collection of images or calexp's to an RGB image
    
    This requires either an array of images or a list of calexps.
    If filter indices is not specified, it uses the first three images in opposite order
    (for example if images=[g, r, i], i->R, r->G, g->B).
    xRange and yRange can be passed to slice an image.
    """
    if images is None:
        if calexps is None:
            raise ValueError("You must either pass an array of images or set of calexps")
        images = np.array([calexp.getMaskedImage().getImage().getArray() for calexp in calexps])
    if len(images)<3:
        raise ValueError("Expected either an array of 3 or more images or a list of 3 or more calexps")
    if filterIndices is None:
        filterIndices = [2,1,0]
    if yRange is None:
        ySlice = slice(None, None)
    elif not isinstance(yRange, slice):
        ySlice = slice(yRange[0], yRange[1])
    else:
        ySlice = yRange
    if xRange is None:
        xSlice = slice(None, None)
    elif not isinstance(xRange, slice):
        xSlice = slice(xRange[0], xRange[1])
    else:
        xSlice = xRange
    # Select the subset of 3 images to use for the RGB image
    images = images[filterIndices,ySlice, xSlice]

    # Map intensity to [0,255]
    intensity = np.arcsinh(contrast*np.sum(images, axis=0)/3)
    if adjustZero:
        # Adjust the colors so that zero is the lowest flux value
        intensity = (intensity-np.min(intensity))/(np.max(intensity)-np.min(intensity))*255
    else:
        intensity = intensity/(np.max(intensity))*255
        intensity[intensity<0] = 0

    # Use the absolute value to normalize the pixel intensities
    pixelIntensity = np.sum(np.abs(images), axis=0)
    # Prevent division by zero
    zeroPix = pixelIntensity==0
    pixelIntensity[zeroPix] = 1

    # Calculate the RGB colors
    pixelIntensity = np.broadcast_to(pixelIntensity, (3, pixelIntensity.shape[0], pixelIntensity.shape[1]))
    intensity = np.broadcast_to(intensity, (3, intensity.shape[0], intensity.shape[1]))
    zeroPix = np.broadcast_to(zeroPix, (3, zeroPix.shape[0], zeroPix.shape[1]))
    colors = images/pixelIntensity*intensity
    colors[colors<0] = 0
    colors[zeroPix] = 0
    colors = colors.astype(np.uint8)
    return np.dstack(colors)

def plotColorImage(images=None, calexps=None, filterIndices=None, xRange=None, yRange=None,
                   contrast=100, adjustZero=True, figsize=(5,5)):
    """Display a collection of images or calexp's as an RGB image
    
    See `imagesToRgb` for more info.
    """
    colors = imagesToRgb(images, calexps, filterIndices, xRange, yRange, contrast, adjustZero)
    plt.figure(figsize=figsize)
    plt.imshow(colors)
    plt.show()
    return colors

def maskPlot(img, mask=None, hideAxes=True, show=True, **kwargs):
    """Plot an image with specified pictures masked out
    
    It is often convenient to mask zero (or low flux) pixels in an image to highlight actual structure,
    so this convenience function makes it quick and easy to implement
    ``plt.plot(np.ma.array(img, mask=mask), **kwargs)``, optionally hiding the axes and showing the image.
    """
    if mask is None:
        maImg = img
    else:
        maImg = np.ma.array(img, mask=mask)
    plt.imshow(maImg, **kwargs)
    if hideAxes:
        plt.axis("off")
    if show:
        plt.show()
    return plt