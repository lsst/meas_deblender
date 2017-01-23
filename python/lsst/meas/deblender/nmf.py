# Temporary file to test the NMF algorithm

from collections import OrderedDict
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table as ApTable

import lsst.log as log
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
from lsst.meas.deblender.baseline import newDeblend
from lsst.meas.deblender import plugins as debPlugins

logger = logging.getLogger()

def get_footprint_arr(src):
    """Get the border and filled in arrays of a footprint
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
    
    if filled.shape[0] ==0:
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
    """Calculate minimum and maximum pixel values
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

def get_peak_footprint(peak):
    """Get the border of a footprint for a given peak as an array, with its bounding box.
    """
    template_footprint = peak.templateFootprint
    bbox = template_footprint.getBBox()
    minX = bbox.getMinX()
    minY = bbox.getMinY()
    maxX = bbox.getMaxX()
    maxY = bbox.getMaxY()

    border, filled = get_footprint_arr(template_footprint)
    return border, minX, minY, maxX, maxY

def DEPRECATEDextract_deblended_img(src, img, x0, y0):
    footprint = src.getFootprint()
    bbox = footprint.getBBox()
    minX = bbox.getMinX()
    minY = bbox.getMinY()
    maxX = bbox.getMaxX()
    maxY = bbox.getMaxY()
    img_footprint = img[minY-y0:maxY-y0, minX-x0:maxX-x0]
    return img_footprint, minX, minY, maxX, maxY

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

def buildNmfData(calexps, footprint):
    """Build NMF data matrix
    
    Given an ordered dict of exposures in each band,
    create a matrix with rows as the image pixels in each band.
    
    Eventually we will also want to mask pixels, but for now we ignore masking.
    """
    # Since these are calexps, they should all have the same x0, y0 (initial pixel positions)
    f = calexps.keys()[0]
    x0 = calexps[f].getX0()
    y0 = calexps[f].getY0()
    
    bbox = footprint.getBBox()
    xmin = bbox.getMinX()-x0
    xmax = xmin+bbox.getWidth()
    ymin = bbox.getMinY()-y0
    ymax = ymin+bbox.getHeight()
    
    bandCount = len(calexps)
    pixelCount = (ymax-ymin)*(xmax-xmin)
    data = np.zeros((bandCount, pixelCount), dtype=np.float64)
    mask = np.zeros((bandCount, pixelCount), dtype=np.int64)
    
    for n, (f,calexp) in enumerate(calexps.items()):
        img, m, var = calexp.getMaskedImage().getArrays()
        data[n] = img[ymin:ymax, xmin:xmax].flatten()
        mask[n] = m[ymin:ymax, xmin:xmax].flatten()
    
    return data, mask

def createInitWH(debResult, footprint, filters, includeBkg=True):
    """Use the deblender results to estimate the initial conditions for intensity of the peaks
    """
    bbox = footprint.getBBox()
    peakCount = len(debResult.peaks)
    if includeBkg:
        peakCount += 1
    H = np.zeros((peakCount,bbox.getArea()))
    W = np.zeros((len(debResult.filters), peakCount))
    for p,pk in enumerate(debResult.peaks):
        fpPixels = np.zeros((len(debResult.filters),bbox.getArea()))
        for n,f in enumerate(filters):
            peak = pk.deblendedPeaks[f]
            bbox = footprint.getBBox()
            Htemplate = np.zeros((bbox.getHeight(), bbox.getWidth()))
            xslice, yslice = getRelativeSlices(peak.templateFootprint.getBBox(), bbox)
            Htemplate[yslice, xslice] = peak.templateImage.getArray()
            fpPixels[n,:] = Htemplate.flatten()
        sed = np.max(fpPixels, axis=1)
        fpPixels = (fpPixels.T / sed).T
        H[p,:] = np.sum(fpPixels, axis=0)/np.sum(sed)
        W[:,p] = sed
    if includeBkg:
        W[:,-1] = 0
        pkIntensity = np.sum(H[:-1,:], axis=0)
        H[-1,:][pkIntensity<1e-2] = 1
    return W, H

def plotSeds(W):
    """Plot the SEDs for each source
    """
    for col in range(W.shape[1]):
        sed = W[:, col]
        band = range(len(sed))
        plt.plot(band, sed, '.-')
    plt.xlabel("Filter Number")
    plt.ylabel("Flux")
    plt.show()

def plotIntensities(H, W, fp, fidx=0, vmin=None, vmax=None, showBkg=False):
    """Plot the template image for each source
    """
    t=[]
    N = H.shape[0]
    if not showBkg:
        N = N-1
    for n in range(N):
        template = W[fidx,n]*H[n,:].reshape(fp.getBBox().getHeight(),fp.getBBox().getWidth())
        if n<H.shape[0]-1:
            template = np.ma.array(template, mask=template==0)
        plt.imshow(template, interpolation='none', cmap='inferno', vmin=vmin, vmax=vmax)
        plt.show()
        t.append(template)

def getPeakSymmetryOperator(row, shape, px, py):
    """Build the operator to symmetrize a single row in H, the intensities of a single peak.
    """
    center = (np.array(shape)-1)/2.0
    # If the peak is centered at the middle of the footprint,
    # make the entire footprint symmetric
    if px==center[1] and py==center[0]:
        return np.fliplr(np.eye(np.size(row)))
    
    if py<(shape[0]-1)/2.:
        ymin = 0
        ymax = 2*py+1
    elif py>(shape[0]-1)/2.:
        ymin = 2*py-shape[0]+1
        ymax = shape[0]
    else:
        ymin = 0
        ymax = shape[0]
    if px<(shape[1]-1)/2.:
        xmin = 0
        xmax = 2*px+1
    elif px>(shape[1]-1)/2.:
        xmin = 2*px-shape[1]+1
        xmax = shape[1]
    else:
        xmin = 0
        xmax = shape[1]
    
    fpHeight, fpWidth = shape
    fpSize = fpWidth*fpHeight
    tWidth = xmax-xmin
    tHeight = ymax-ymin
    
    extraWidth = fpWidth-tWidth
    pixels = (tHeight-1)*fpWidth+tWidth
    subOp = np.eye(pixels,pixels)
    
    for i in range(0,tHeight-1):
        for j in range(extraWidth):
            idx = (i+1)*tWidth+(i*extraWidth)+j
            subOp[idx, idx] = 0
    
    subOp = np.fliplr(subOp)
    smin = ymin*fpWidth+xmin
    smax = (ymax-1)*fpWidth+xmax
    symmetryOp = np.zeros((fpSize, fpSize))
    #symmetryOp[:pixels,:pixels] = subOp
    symmetryOp[smin:smax,smin:smax] = subOp
    
    return symmetryOp

def getSymmetryOperator(H, fp):
    """Build the symmetry operator for an entire intensity matrix H
    """
    symmetryOp = []
    bbox = fp.getBBox()
    fpShape = (bbox.getHeight(),bbox.getWidth())
    for n,peak in enumerate(fp.getPeaks()):
        px = peak.getIx()-fp.getBBox().getMinX()
        py = peak.getIy()-fp.getBBox().getMinY()
        s = getPeakSymmetryOperator(H[n:], fpShape, px, py)
        symmetryOp.append(s)
    #bkg = H[-1,:].reshape(fp.getBBox().getHeight(),fp.getBBox().getWidth())
    #Hsym[-1,:] = np.flipud(np.fliplr(bkg)).flatten()
    return symmetryOp

def getSymmetryDiffOp(H, fp, includeBkg=False):
    """Build the symmetry difference Operator
    
    This creates the pseudo operator $(I-S)^2$, where I is the identity matrix
    and S is the symmetry operator for each row k in H.
    """
    symmetryObj = getSymmetryOperator(H, fp)
    diffOp = []
    for k, sk in enumerate(symmetryObj):
        diff = np.eye(sk.shape[k]) + np.dot(sk,sk) - 2*sk
        diffOp.append(diff)
    if includeBkg:
        diffOp.append(np.eye(len(H[-1])))
    return diffOp

def getIntensityDiff(H, diffOp):
    """Calculate the cost function penalty for non-symmetry
    
    Given a pre-calculated symmetry difference operator, calculate the differential term in
    the cost function to penalize for non-symmetry
    """
    Hdiff = np.zeros(H.shape)
    for k, Hk in enumerate(H):
        Hdiff[k] = np.dot(diffOp[k], Hk)
    return Hdiff

def nmfUpdate(A, W, H, beta, fp, diffOp=None):
    """Update the SED (W) and intensity (H) matrices
    """
    numerator = np.matmul(A, H.T)
    denom = np.maximum(np.matmul(np.matmul(W,H), H.T),1e-9)
    W = W * numerator/denom
    
    numerator = np.matmul(W.T,A)
    if diffOp is not None:
        Hdiff = getIntensityDiff(H, diffOp)
        denom = np.matmul(np.matmul(W.T,W), H)+beta*Hdiff + 1e-9
    else:
        denom = np.matmul(np.matmul(W.T,W), H) + 1e-9
    H = H * numerator/denom
    return W, H

def loadCalExps(filters, filename):
    """Load calexps for testing the deblender.
    
    This function is only for testing and will be removed before merging.
    Given a list of filters and a filename template, load a set of calibrated exposures.
    """
    calexps = OrderedDict()
    tables = OrderedDict()
    vminDict = OrderedDict()
    vmaxDict = OrderedDict()
    for f in filters:
        logging.info("Loading filter {0}".format(f))
        calexps[f] = afwImage.ExposureF(filename.format("calexp",f))
        vminDict[f], vmaxDict[f] = zscale(calexps[f].getMaskedImage().getImage().getArray())
    x0 = calexps[filters[0]].getX0()
    y0 = calexps[filters[0]].getY0()
    return calexps, tables, vminDict, vmaxDict

def loadMergedDetections(filename):
    """Load mergedDet catalog ``filename``
    
    This function is for testing only and will be removed before merging.
    """
    mergedDet = afwTable.SourceCatalog.readFits(filename)
    columns = []
    names = []
    for col in mergedDet.getSchema().getNames():
        names.append(col)
        columns.append(mergedDet.columns.get(col))
    columns.append([len(src.getFootprint().getPeaks()) for src in mergedDet])
    names.append("peaks")
    mergedTable = ApTable(columns, names=tuple(names))

    logging.info("Total parents: {0}".format(len(mergedTable)))
    logging.info("Unblended sources: {0}".format(np.sum(mergedTable['peaks']==1)))
    logging.info("Sources with multiple peaks: {0}".format(np.sum(mergedTable['peaks']>1)))
    return mergedDet, mergedTable

def loadSimCatalog(filename):
    """Load a catalog of galaxies generated by galsim
    """
    cat = afwTable.BaseCatalog.readFits(filename)
    columns = []
    names = []
    for col in cat.getSchema().getNames():
        names.append(col)
        columns.append(cat.columns.get(col))
    catTable = ApTable(columns, names=tuple(names))
    return cat, catTable

def getParentFootprint(mergedTable, mergedDet, calexps, condition, parentIdx, filt, display=True, **kwargs):
    """Load the parent footprint and peaks, and (optionally) display the image and footprint border
    """
    idx = np.where(condition)[0][parentIdx]
    src = mergedDet[idx]
    fp = src.getFootprint()
    bbox = fp.getBBox()
    peaks = fp.getPeaks()
    
    if "interpolation" not in kwargs:
        kwargs["interpolation"] = 'none'
    if "cmap" not in kwargs:
        kwargs["cmap"] = "inferno"
    
    if display:
        img = extractImage(calexps[filt].getMaskedImage(), bbox)
        plt.imshow(img, **kwargs)
        border, filled = get_footprint_arr(src)
        plt.imshow(border, interpolation='none', cmap='cool')

        px = [peak.getIx()-bbox.getMinX() for peak in fp.getPeaks()]
        py = [peak.getIy()-bbox.getMinY() for peak in fp.getPeaks()]
        plt.plot(px, py, "rx")
        plt.xlim(0,img.shape[1]-1)
        plt.ylim(0,img.shape[0]-1)
        plt.show()
    return fp, peaks

def initNMFparams(calexps, fp, filters, filterIdx=0):
    """Create the initial data, SED, and Intensity matrices using the current deblender
    
    The SED estimate is the value of the peak in each filter, and the intensity is the normalized
    template for each source.
    """
    plugins = [debPlugins.DeblenderPlugin(debPlugins.buildSymmetricTemplates)]
    footprints = [fp]*len(filters)
    maskedImages = [calexp.getMaskedImage() for f, calexp in calexps.items()]
    psfs = [calexp.getPsf() for f, calexp in calexps.items()]
    fwhm = [psf.computeShape().getDeterminantRadius() * 2.35 for psf in psfs]
    debResult = newDeblend(plugins, footprints, maskedImages, psfs, fwhm, filters=filters)
    data, mask = buildNmfData(calexps, fp)
    W, H = createInitWH(debResult, fp, filters)
    return data, mask, W, H, debResult

def alsDeblend(fp, data, W, H, fidx, beta, includeBkg=False, display=True, verbose=True):
    """Use ALS to factorize the data into an SED matrix and Intensity matrix
    
    Currently this is experimental. One drawback to this method is that (for now) it does not
    use the symmetry constraint, which still gives reasonable results 
    """
    if includeBkg:
        newW = np.copy(W)
        newH = np.copy(H)
    else:
        newW = np.copy(W[:,:-1])
        newH = np.copy(H[:-1,:])
    fpShape = (fp.getBBox().getHeight(), fp.getBBox().getWidth())
    for i in range(20):
        newH = np.dot(np.linalg.inv(np.dot(newW.T, newW)+1e-9), np.dot(newW.T, data))
        newW = np.dot(np.linalg.inv(np.dot(newH, newH.T)+1e-9), np.dot(newH, data.T)).T
    diff = (np.dot(newW, newH)-data)[fidx].reshape(fpShape)
    if verbose:
        logger.info('Pixel range: {0} to {1}'.format(np.min(data), np.max(data)))
        logger.info('Max difference: {0}'.format(np.max(diff)))
        logger.info('Residual difference {0:.1f}%'.format(100*np.abs(np.sum(diff)/np.sum(data[fidx]))))
    if display:
        plotIntensities(newH, newW, fp, fidx, showBkg=True)
        plotSeds(newW)
        plt.imshow(diff, interpolation='none', cmap='inferno')
        plt.show()
    return newW, newH

def multiplicativeDeblend(fp, data, W, H, fidx, beta, includeBkg=True, display=True, verbose=True):
    if includeBkg:
        newW = np.copy(W)
        newH = np.copy(H)
    else:
        newW = np.copy(W[:,:-1])
        newH = np.copy(H[:-1,:])
    diffOp = getSymmetryDiffOp(newH, fp, includeBkg=includeBkg)
    fpShape = (fp.getBBox().getHeight(), fp.getBBox().getWidth())

    for i in range(200):
        newW, newH = nmfUpdate(data, newW, newH, beta, fp, diffOp)

    diff = (np.dot(newW, newH)-data)[fidx].reshape(fpShape)
    if verbose:
        logger.info('Pixel range: {0} to {1}'.format(np.min(data), np.max(data)))
        logger.info('Max difference: {0}'.format(np.max(diff)))
        logger.info('Residual difference {0:.1f}%'.format(100*np.abs(np.sum(diff)/np.sum(data[fidx]))))
    if display:
        plotIntensities(newH, newW, fp, fidx, showBkg=True)
        plotSeds(newW)
        plt.imshow(diff, interpolation='none', cmap='inferno')
        plt.show()
    return newW, newH