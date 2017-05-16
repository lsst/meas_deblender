# Utilities useful for developing the deblender.
# This file is not expected to be merged to master, although subsets of it might be ported later

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def getFootprintArray(src):
    """Get the border and filled in arrays of a footprint

    Extracting the mask is currently implemented in ``Footprints``, but since this functionality has
    been moved to ``SpanSet``s we will fix it later.
    """
    if hasattr(src, "getFootprint"):
        footprint = src.getFootprint()
    else:
        footprint = src
    spans = footprint.getSpans()
    bbox = footprint.getBBox()
    minX = bbox.getMinX()
    minY = bbox.getMinY()
    filled = np.ma.array(np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool))
    border = np.ma.array(np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool))

    if filled.shape[0]==0:
        return border, filled

    for n,span in enumerate(spans):
        y = span.getY();
        filled[y-minY, span.getMinX()-minX:span.getMaxX()-minX] = 1
        border[y-minY,span.getMinX()-minX] = 1
        border[y-minY,span.getMaxX()-minX-1] = 1
    border[0] = filled[0]
    border[-1] = filled[-1]
    for n,row in enumerate(border[:-1]):
        border[n] = border[n] | (filled[n] & ((filled[n]^filled[n-1])|(filled[n]^filled[n+1])))
    border.mask = ~border[:]
    filled.mask = ~filled[:]
    return border, filled

def zscale(img, contrast=0.25, samples=500):
    """Calculate minimum and maximum pixel values based on the image

    From RHL, via Bob
    """
    ravel = img.ravel()
    if len(ravel) > samples:
        imsort = np.sort(np.random.choice(ravel, size=samples))
    else:
        imsort = np.sort(ravel)

    n = len(imsort)
    idx = np.arange(n)

    med = imsort[int(n/2)]
    w = 0.25
    i_lo, i_hi = int((0.5-w)*n), int((0.5+w)*n)
    p = np.polyfit(idx[i_lo:i_hi], imsort[i_lo:i_hi], 1)
    slope, intercept = p

    z1 = med - (slope/contrast)*(n/2-n*w)
    z2 = med + (slope/contrast)*(n/2-n*w)

    return z1, z2

def getRelativeSlices(bbox, refBbox):
    """Get the slice to clip an image from its bounding box
    """
    xmin = bbox.getMinX()-refBbox.getMinX()
    ymin = bbox.getMinY()-refBbox.getMinY()
    xmax = xmin+bbox.getWidth()
    ymax = ymin+bbox.getHeight()
    return slice(xmin, xmax), slice(ymin, ymax)

def extractImage(img, bbox):
    """Extract an array from a maskedImage based on the bounding box
    """
    refBbox = img.getBBox()
    xslice, yslice = getRelativeSlices(bbox, refBbox)
    return img.getImage().getArray()[yslice, xslice]
