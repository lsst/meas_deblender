# Temporary file to test NMF algorithms.
# Some of these functions and classes are likely to remain if
# NMF is a viable solution.
# Otherwise the entire module will be junked without merging to master
from __future__ import print_function, division
from collections import OrderedDict
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from astropy.table import Table as ApTable

import lsst.log as log
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
from .baseline import newDeblend
from . import plugins as debPlugins
from . import utils as debUtils

logging.basicConfig()
logger = logging.getLogger("lsst.meas.deblender")

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
    
    # Add the image in each filter as a row in data
    data = np.zeros((bandCount, pixelCount), dtype=np.float64)
    mask = np.zeros((bandCount, pixelCount), dtype=np.int64)
    for n, (f,calexp) in enumerate(calexps.items()):
        img, m, var = calexp.getMaskedImage().getArrays()
        data[n] = img[ymin:ymax, xmin:xmax].flatten()
        mask[n] = m[ymin:ymax, xmin:xmax].flatten()
    
    return data, mask

def createInitWH(data, debResult, footprint, includeBkg=True, fillZero=False, normalize=True):
    """Use the deblender symmetric templates to estimate the initial conditions for intensity of the peaks
    """
    bbox = footprint.getBBox()
    peakCount = len(debResult.peaks)
    if includeBkg:
        peakCount += 1
    H = np.zeros((peakCount,bbox.getArea()))
    W = np.zeros((len(debResult.filters), peakCount))
    for p,pk in enumerate(debResult.peaks):
        fpPixels = np.zeros((len(debResult.filters),bbox.getArea()))
        for n,f in enumerate(debResult.filters):
            peak = pk.deblendedPeaks[f]
            bbox = footprint.getBBox()
            Htemplate = np.zeros((bbox.getHeight(), bbox.getWidth()))
            xslice, yslice = debUtils.getRelativeSlices(peak.templateFootprint.getBBox(), bbox)
            Htemplate[yslice, xslice] = peak.templateImage.getArray()
            fpPixels[n,:] = Htemplate.flatten()
        H[p,:] = np.mean(fpPixels, axis=0)
    # If a multiplicative algorithm is used, pixels with zero intensity will never change
    # Introducing a very small non-zero value outside the footprint allows the algorithm slightly
    # more flexibility
    # However, when offsetting the data this is not important as the zero offset values can be
    # used in the iteration
    if fillZero:
        H[H==0] = 1e-9
    # Solve for W using the pseudo inverse, in case H is singular
    W = np.dot(data, np.linalg.pinv(H))
    
    # Normalize the SEDs to one and re-calculate H
    if normalize:
        normFactor = np.sum(W, axis=0)
        # Adjust the norm factor to not divide by zero (even though it is dividing zero by zero)
        normFactor[normFactor==0] = 1
        W = W / normFactor
        H = (H.T * normFactor).T
    
    # It is possible to create a separate object to dump all of the background pixels.
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

def reconstructTemplate(W, H, offset, fidx , pkIdx, fp=None, reshape=False):
    WH = np.dot(W,H)
    WHk = W[fidx,pkIdx]*H[pkIdx,:]
    # Fraction of each pixel in the template for source k
    sumWH = WH[pkIdx,:]
    sumWH[sumWH==0] = 1 # To avoid dividing 0/0
    fraction = WHk/sumWH
    # Fraction of the offset to subtract from the pixels in source k
    offsetMatrix = offset*fraction
    template = WHk-offsetMatrix
    if reshape:
        if fp is None:
            raise ValueError("You must pass a footprint to reshape the template")
        template = template.reshape(fp.getBBox().getHeight(),fp.getBBox().getWidth())
    return template

def plotIntensities(W, H, fp, offset=0, fidx=0, vmin=None, vmax=None, useMask=False):
    """Plot the template image for each source
    
    Multiply each row in H (pixel intensities for a single source) by the SED for filter ``fidx`` and
    plot the result.
    """
    for k in range(len(H)):
        template = reconstructTemplate(W, H, offset, fidx, k, fp, reshape=True)
        # Mask zero pixels (only useful when not using an offset)
        if useMask and k<templates.shape[0]-1:
            template = np.ma.array(template, mask=template==0)
        plt.imshow(template, interpolation='none', cmap='inferno', vmin=vmin, vmax=vmax)
        plt.show()

def loadCalExps(filters, filename):
    """Load calexps for testing the deblender.
    
    This function is only for testing and will be removed before merging.
    Given a list of filters and a filename template, load a set of calibrated exposures.
    """
    calexps = OrderedDict()
    vminDict = OrderedDict()
    vmaxDict = OrderedDict()
    for f in filters:
        logger.info("Loading filter {0}".format(f))
        calexps[f] = afwImage.ExposureF(filename.format("calexp",f))
        vminDict[f], vmaxDict[f] = debUtils.zscale(calexps[f].getMaskedImage().getImage().getArray())
    x0 = calexps[filters[0]].getX0()
    y0 = calexps[filters[0]].getY0()
    return calexps, vminDict, vmaxDict

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

    logger.info("Total parents: {0}".format(len(mergedTable)))
    logger.info("Unblended sources: {0}".format(np.sum(mergedTable['peaks']==1)))
    logger.info("Sources with multiple peaks: {0}".format(np.sum(mergedTable['peaks']>1)))
    return mergedDet, mergedTable

def loadSimCatalog(filename):
    """Load a catalog of galaxies generated by galsim
    
    This can be used to ensure that the deblender is correctly deblending objects
    """
    cat = afwTable.BaseCatalog.readFits(filename)
    columns = []
    names = []
    for col in cat.getSchema().getNames():
        names.append(col)
        columns.append(cat.columns.get(col))
    catTable = ApTable(columns, names=tuple(names))
    return cat, catTable

def getParentFootprint(mergedTable, mergedDet, calexps, condition, parentIdx, display=True, filt=None,
        **kwargs):
    """Load the parent footprint and peaks, and (optionally) display the image and footprint border
    """
    idx = np.where(condition)[0][parentIdx]
    src = mergedDet[idx]
    fp = src.getFootprint()
    bbox = fp.getBBox()
    peaks = fp.getPeaks()
    
    if display:
        if "interpolation" not in kwargs:
            kwargs["interpolation"] = 'none'
        if "cmap" not in kwargs:
            kwargs["cmap"] = "inferno"
        
        img = debUtils.extractImage(calexps[filt].getMaskedImage(), bbox)
        plt.imshow(img, **kwargs)
        border, filled = debUtils.getFootprintArray(src)
        plt.imshow(border, interpolation='none', cmap='cool')

        px = [peak.getIx()-bbox.getMinX() for peak in fp.getPeaks()]
        py = [peak.getIy()-bbox.getMinY() for peak in fp.getPeaks()]
        plt.plot(px, py, "rx")
        plt.xlim(0,img.shape[1]-1)
        plt.ylim(0,img.shape[0]-1)
        plt.show()
    return fp, peaks

def initNMFparams(calexps, fp, fillZero=False, includeBkg=True, normalize=True):
    """Create the initial data, SED, and Intensity matrices using the current deblender
    
    The SED estimate is the value of the peak in each filter, and the intensity is the normalized
    template for each source.
    """
    plugins = [debPlugins.DeblenderPlugin(debPlugins.buildSymmetricTemplates)]
    footprints = [fp]*len(calexps.keys())
    maskedImages = [calexp.getMaskedImage() for f, calexp in calexps.items()]
    psfs = [calexp.getPsf() for f, calexp in calexps.items()]
    fwhm = [psf.computeShape().getDeterminantRadius() * 2.35 for psf in psfs]
    debResult = newDeblend(plugins, footprints, maskedImages, psfs, fwhm, filters=calexps.keys())
    data, mask = buildNmfData(calexps, fp)
    
    #TODO: fix offsets
    #data[data<0] = 0
    if np.sum(data<0) > 0:
        offset = -np.min(data)
        data = data + offset
    else:
        offset=0
    W, H = createInitWH(data, debResult, fp, fillZero=fillZero, includeBkg=includeBkg, normalize=normalize)
    return data, mask, W, H, debResult, offset

def getPeakSymmetryOperator(row, shape, px, py):
    """Build the operator to symmetrize a single row in H, the intensities of a single peak.
    """
    center = (np.array(shape)-1)/2.0
    # If the peak is centered at the middle of the footprint,
    # make the entire footprint symmetric
    if px==center[1] and py==center[0]:
        return np.fliplr(np.eye(np.size(row)))
    
    # Otherwise, find the bounding box that contains the minimum number of pixels needed to symmetrize
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
    
    # This is the block of the matrix that symmetrizes intensities at the peak position
    subOp = np.eye(pixels, pixels)
    for i in range(0,tHeight-1):
        for j in range(extraWidth):
            idx = (i+1)*tWidth+(i*extraWidth)+j
            subOp[idx, idx] = 0
    subOp = np.fliplr(subOp)
    
    smin = ymin*fpWidth+xmin
    smax = (ymax-1)*fpWidth+xmax
    symmetryOp = np.zeros((fpSize, fpSize))
    symmetryOp[smin:smax,smin:smax] = subOp
    
    # Return a sparse matrix, which greatly speeds up the processing
    return scipy.sparse.coo_matrix(symmetryOp)

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
    return symmetryOp

def getSymmetryDiffOp(H, fp, includeBkg=False):
    """Build the symmetry difference Operator
    
    This creates the pseudo operator $(I-S)^2$, where I is the identity matrix
    and S is the symmetry operator for each row k in H.
    """
    symmetryObj = getSymmetryOperator(H, fp)
    diffOp = []
    for k, sk in enumerate(symmetryObj):
        diff = scipy.sparse.eye(sk.shape[0]) + sk.dot(sk) - 2*sk
        diffOp.append(diff)
    if includeBkg:
        diffOp.append(scipy.sparse.eye(len(H[-1])))
    return diffOp

def getIntensityDiff(H, diffOp):
    """Calculate the cost function penalty for non-symmetry
    
    Given a pre-calculated symmetry difference operator, calculate the differential term in
    the cost function to penalize for non-symmetry
    """
    Hdiff = np.zeros(H.shape)
    for k, Hk in enumerate(H):
        Hdiff[k] = diffOp[k].dot(Hk)
    return Hdiff

def multiplicativeUpdate(A, W, H, fp, alpha=0, beta=0, diffOp=None, normalize=True):
    """Update the SED (W) and intensity (H) matrices using Lee and Seung 2001
    
    Use the Lee and Seung multiplicative algorithm, which is basically a gradient descent with
    step sizes that change based on the current W,H values, guaranteeing that W,H are always positive.
    
    For now, alpha is the penalty for the W smoothness constraint and beta is the penalty for symmetry in H
    """
    numerator = np.matmul(A, H.T)
    denom = np.matmul(np.matmul(W,H), H.T) + 1e-9 + alpha*W
    W = W * numerator/denom
    if normalize:
        normFactor = np.sum(np.abs(W), axis=0)
        # If a column is all zero (for example the background), we can't divide by zero
        normFactor[normFactor==0] = 1
        W = W/normFactor
    
    numerator = np.matmul(W.T,A)
    if diffOp is not None:
        Hdiff = getIntensityDiff(H, diffOp)
        numerator = np.matmul(W.T,A)
        denom = np.matmul(np.matmul(W.T,W), H) + beta*Hdiff + 1e-9
    else:
        denom = np.matmul(np.matmul(W.T,W), H) + 1e-9
    H = H * numerator/denom
    
    return W, H

def inverseUpdate(A, W, H, fp):
    """Exactly solve A=WH for W,H using an inverse.
    
    Currently this method has no constraints, as operations like symmetry and monotonicity are non-linear.
    """
    newH = np.dot(np.linalg.inv(np.dot(W.T, W)+1e-9), np.dot(W.T, A))
    newW = np.dot(np.linalg.inv(np.dot(newH, newH.T)+1e-9), np.dot(newH, A.T)).T
    return newW, newH

def gradientDescent(A, W, H, fp, stepW=.001, stepH=.001, alpha=0, beta=0, diffOp=None):
    """Use gradient descent to update W,H
    """
    WH = np.dot(W,H)
    
    derivH = np.dot(W.T, -A+WH)
    if diffOp is not None:
        Hdiff = getIntensityDiff(H, diffOp)
        derivH = derivH + beta*Hdiff
    H = H - stepH*derivH
    
    derivW = np.dot(-A+WH, H.T)
    W = W - stepW*derivW
    stepW = stepW/2.
    stepH = stepH/2.
    return W, H

def compareMeasToSim(fp, W, H, realTable, filters, offset=0, vminDict=None, vmaxDict=None,
                     display=False, ignoreNegative=True):
    """Compare measurements to simulated "true" data
    
    If running nmf on simulations, this matches the detections to the simulation catalog and
    compares the measured flux of each object to the simulated flux.
    """
    peakCoords = np.array([[peak.getIx(),peak.getIy()] for peak in fp.getPeaks()])
    refCoords = np.array(list(zip(realTable['x'],realTable['y'])))
    idx, d2 = debUtils.query_reference(peakCoords, refCoords, kdtree=None, 
            pool_size=None, separation=3, radec=False)
    fpShape = (fp.getBBox().getHeight(), fp.getBBox().getWidth())
    
    for k in range(len(W)-1):
        logger.info("Object {0} at ({1},{2})".format(k, fp.getPeaks()[k].getIx(), fp.getPeaks()[k].getIx()))
        for fidx, f in enumerate(filters):
            template = reconstructTemplate(W, H, offset, fidx , pkIdx=k, fp=fp, reshape=True)
            if ignoreNegative:
                measFlux = np.sum(template[template>0])
            else:
                measFlux = np.sum(template)
            realFlux = realTable[idx][k]['flux_'+f]
            logger.info("Filter {0}: flux={1}, real={2}, error={3:.2f}%".format(
                f, measFlux, realFlux, 100*np.abs(measFlux-realFlux)/realFlux))
            if display:
                kwargs = {}
                if vminDict is not None:
                    kwargs["vmin"] = vminDict[f]
                if vmaxDict is not None:
                    kwargs["vmax"] = vmaxDict[f]*10
                plt.imshow(theory, interpolation='none', cmap='inferno', **kwargs)
                plt.show()
    return realTable[idx]

class MulticolorCalExp:
    """Container for the objects needed for the NMF deblender
    """
    def __init__(self, filters, imgFilename, mergedDetFilename, simFilename=None):
        self.filters = filters
        self.loadFiles(imgFilename, mergedDetFilename, simFilename)
    
    def loadFiles(self, imgFilename=None, mergedDetFilename=None, simFilename=None):
        """Load images in each filter, the merged catalog and (optionally) a sim catalog
        """
        if imgFilename is not None:
            self.imgFilename = imgFilename
            self.calexps, self.vminDict, self.vmaxDict = loadCalExps(self.filters, imgFilename)
        if mergedDetFilename is not None:
            self.mergedDetFilename = mergedDetFilename
            self.mergedDet, self.mergedTable = loadMergedDetections(mergedDetFilename)
        if simFilename is not None:
            self.simFilename = simFilename
            self.simCat, self.simTable = loadSimCatalog(simFilename)
        elif not hasattr(self, "simFilename"):
            self.simFilename = None
    
    def getParentFootprint(self, parentIdx=0, condition=None, display=True, imgLimits=True, **displayKwargs):
        """Get the parent footprint, peaks, and (optionally) display them
        """
        if condition is None:
            condition = slice(0, len(self.mergedTable))
        if display:
            if "filt" not in displayKwargs:
                displayKwargs["filt"] = self.filters[0]
            if imgLimits:
                if "vmin" not in displayKwargs:
                    displayKwargs["vmin"] = self.vminDict[displayKwargs["filt"]]
                if "vmax" not in displayKwargs:
                    displayKwargs["vmax"] = 10*self.vmaxDict[displayKwargs["filt"]]
        self.footprint, self.peaks = getParentFootprint(self.mergedTable, self.mergedDet, self.calexps, 
                                                        condition, parentIdx, display, **displayKwargs)
        return self.footprint, self.peaks
    
    def initNMFParams(self, fillZero=True, normalize=True, displaySeds=True, displayTemplates=True,
                      imgLimits=True, includeBkg=True, **displayKwargs):
        """Initialize the parameters needed for NMF deblending and (optionally) display the results
        """
        result = initNMFparams(self.calexps, self.footprint, fillZero=fillZero,
                               includeBkg=includeBkg, normalize=normalize)
        self.data, self.mask, self.initW, self.initH, self.debResult, self.offset = result
        if displaySeds:
            plotSeds(self.initW)
        if displayTemplates:
            if "fidx" not in displayKwargs:
                displayKwargs["fidx"] = 0
            if imgLimits:
                if "vmin" not in displayKwargs:
                    displayKwargs["vmin"] = self.vminDict[self.filters[displayKwargs["fidx"]]]
                if "vmax" not in displayKwargs:
                    displayKwargs["vmax"] = 10*self.vmaxDict[self.filters[displayKwargs["fidx"]]]
            plotIntensities(self.initW, self.initH, self.footprint, offset=self.offset, **displayKwargs)
        return self.data, self.mask, self.initW, self.initH, self.offset
    
    def getSymmetryDiffOp(self, includeBkg=False):
        """Create the operator to symmetrize each row in H
        """
        self.diffOp = getSymmetryDiffOp(self.initH, self.footprint, includeBkg)
        return self.diffOp
    
    def getTemplate(self, fidx, pkIdx, W=None, H=None):
        if W is None:
            W = self.W
        if H is None:
            H = self.H
        return reconstructTemplate(W, H, self.offset, fidx , pkIdx, fp=self.footprint, reshape=True)
    
    def displayTemplate(self, fidx, pkIdx, W=None, H=None, cmap='inferno', imgLimits=True, **displayKwargs):
        template = self.getTemplate(fidx, pkIdx, W, H)
        if imgLimits:
            if "vmin" not in displayKwargs:
                displayKwargs["vmin"] = self.vminDict[self.filters[fidx]]
            if "vmax" not in displayKwargs:
                displayKwargs["vmax"] = 10*self.vmaxDict[self.filters[fidx]]
        plt.imshow(template, interpolation='none', cmap=cmap, **displayKwargs)
        plt.show()
    
    def deblend(self, nmfUpdateFunc=multiplicativeUpdate, displayKwargs=dict(), steps=200,
                display=True, imgLimits=True, **updateKwargs):
        """Run the NMF deblender
        
        This will always start from self.initW and self.initH, which can be modified before execution
        """
        newW = np.copy(self.initW)
        newH = np.copy(self.initH)
        fp = self.footprint
        fpShape = (fp.getBBox().getHeight(), fp.getBBox().getWidth())

        # Update W and H using the specified update function
        for i in range(steps):
            newW, newH = nmfUpdateFunc(self.data, newW, newH, self.footprint, **updateKwargs)

        # Show information about the fit
        for fidx, f in enumerate(self.debResult.filters):
            diff = (np.dot(newW, newH)-self.data)[fidx].reshape(fpShape)
            logger.info('Filter {0}'.format(f))
            logger.info('Pixel range: {0} to {1}'.format(np.min(self.data), np.max(self.data)))
            logger.info('Max difference: {0}'.format(np.max(diff)))
            logger.info('Residual difference {0:.1f}%'.format(
                100*np.abs(np.sum(diff)/np.sum(self.data[fidx]))))
        if self.simFilename is not None:
            compareMeasToSim(fp, newW, newH, self.simTable, self.filters, self.offset, display=False)
        
        # Show the new templates for each object
        if display:
            if "fidx" not in displayKwargs:
                displayKwargs["fidx"] = 0
            if imgLimits:
                if "vmin" not in displayKwargs:
                    displayKwargs["vmin"] = self.vminDict[self.filters[displayKwargs["fidx"]]]
                if "vmax" not in displayKwargs:
                    displayKwargs["vmax"] = 10*self.vmaxDict[self.filters[displayKwargs["fidx"]]]
            plotIntensities(newW, newH, fp, offset=self.offset, **displayKwargs)
            plotSeds(newW)
            plt.imshow(diff, interpolation='none', cmap='inferno')
            plt.show()
        
        self.W = newW
        self.H = newH
        return newW, newH
