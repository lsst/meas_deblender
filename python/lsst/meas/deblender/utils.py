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

    med = imsort[n/2]
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

def query_reference(coords, ref_coords=None, kdtree=None, 
        pool_size=None, separation=1/3600, radec=True):
    """
    Get the indices to merge two sets of coordinates together. This includes
    the incides to match both sets, indices of the unmatched rows, and 
    indices of rows that had multiple matches.
    
    Parameters
    ----------
    coords: array-like
        A 2D array with either 2 columns of coordinates (coord1, coord2),
        where coord1 and coord2 are Nx1 arrays, or
        N rows of coordinate pairs 
        [(coord1_1, coord1_2),(coord2_1,coord2_2),...]
        where coordN_1 and coordN_2 are floats.
    ref_coords: array-like, optional
        A 2D array with either 2 columns of coordinates or N rows of
        coordinate pairs (see ``coords`` for more). 
        Either ``ref_coords`` or ``kdtree`` must be specified.
    kdtree: `spatial.cKDTREE`, optional
        KDTree of the reference coordinates (this is an object to use
        for matching positions of objects in k-dimensions, in 2 dimensions
        it is a quad-tree).
        Either ``ref_coords`` or ``kdtree`` must be specified.
    pool_size: int, optional
        Number of processors to use to match coordinates. If 
        ``pool_size is None`` (default) then the maximum number
        of processors is used.
    separation: float, optional
        Maximum distance between two coordinates for a match. The default
        is ``1/3600``, or 1 arcsec.
    radec: bool, optional
        Whether or not the coordinates are given in ra and dec
    
    Returns
    -------
    ref_indices: tuple(idx1, unmatched1)
        Indices to match the reference coordinates to the observed
        coordinates. So the rows of ref_coords[idx1]~coords[idx2],
        ref_coords[unmatched1]~coords[unmatched2], if
        ref_coords and coords are Nx2 arrays of coordinate pairs;
    coord_indices: tuple(idx2, unmatched2)
        Indices to match the coordinates to the reference coordinates.
    duplicates: array-like
        Indices of coordinates with multiple matches, so that
        ref_coords[idx1][duplicates]~coords[idx2][duplicates]
    """
    # If the user didn't specify a KDTREE, 
    if kdtree is None:
        try:
            from scipy import spatial
        except ImportError:
            raise ImportError(
                "You must have 'scipy' installed to combine catalogs")
        if ref_coords is not None:
            if len(ref_coords)==2:
                ref1,ref2 = ref_coords
                pos1 = np.array([ref1,ref2])
                pos1 = pos1.T
            elif len(ref_coords[0])==2:
                pos1 = ref_coords
            else:
                raise ValueError(
                    "Expected either a 2xN array or Nx2 array for ref_coords")
            KDTree = spatial.cKDTree
            kdtree = KDTree(pos1)
        else:
            raise Exception("Either ref_coords or kdtree must be specified")
    if pool_size is None:
        pool_size = -1
    
    if len(coords)==2:
        coord1, coord2 = coords
        pos2 = np.array([coord1,coord2])
        pos2 = pos2.T
    elif len(coords[0])==2:
        pos2 = coords
    else:
        raise ValueError("Expected either a 2xN array or Nx2 array for coords")
    
    # Match all of the sources
    # If using world coordinates, use (ra1-ra2)*cos(dec) for the separation
    if radec:
        d2,idx = kdtree.query(pos2, n_jobs=pool_size)
        # Mask all the rows without a match
        dra = (pos1[:,0][idx]-pos2[:,0])*np.cos(np.deg2rad(pos1[:,1][idx]))
        ddec = pos1[:,1][idx]-pos2[:,1]
        d2 = np.sqrt(dra**2+ddec**2)
        d2[d2>separation] = np.nan
    else:
        d2,idx = kdtree.query(pos2, n_jobs=pool_size,
            distance_upper_bound=separation)
    idx = np.ma.array(idx)
    unmatched = ~np.isfinite(d2)
    idx[unmatched] = -1
    idx.mask = unmatched
    return idx, d2