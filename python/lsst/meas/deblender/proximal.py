# Temporary file to test using proximal operators in the NMF deblender
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
from . import sim

logging.basicConfig()
logger = logging.getLogger("lsst.meas.deblender")

def loadCalExps(filters, filename):
    """Load calexps for testing the deblender.
    
    This function is only for testing and will be removed before merging.
    Given a list of filters and a filename template, load a set of calibrated exposures.
    """
    calexps = []
    vmin = []
    vmax = []
    for f in filters:
        logger.debug("Loading filter {0}".format(f))
        calexps.append(afwImage.ExposureF(filename.format("calexp",f)))
        zscale = debUtils.zscale(calexps[-1].getMaskedImage().getImage().getArray())
        vmin.append(zscale[0])
        vmax.append(zscale[1])
    return calexps, vmin, vmax

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

def getParentFootprint(mergedTable, mergedDet, calexps, condition, parentIdx, display=True, fidx=0,
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
        
        img = debUtils.extractImage(calexps[fidx].getMaskedImage(), bbox)
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

def reconstructTemplate(seds, intensities, fidx , pkIdx, shape=None):
    """Calculate the template for a single peak for a single filter
    
    Use the SED matrix ``seds`` and intensity matrix ``intensities`` to reconstruct the flux in the 
    image due to the selected peak, in the selected filter.
    """
    template = seds[fidx,pkIdx]*intensities[pkIdx,:]
    if shape is not None:
        template = template.reshape(shape)
    return template

def plotIntensities(seds, intensities, shape, fidx=0,
                    vmin=None, vmax=None, useMask=False):
    """Plot the template image for each source
    
    Multiply each row in ``intensities`` by the SED for filter ``fidx`` and
    plot the result.
    """
    for k in range(len(intensities)):
        template = reconstructTemplate(seds, intensities, fidx, k, shape)
        # Optionally Mask zero pixels (gives a better idea of the footprint)
        if useMask:
            template = np.ma.array(template, mask=template==0)
        plt.title("Object {0}".format(k))
        plt.imshow(template, interpolation='none', cmap='inferno', vmin=vmin, vmax=vmax)
        plt.show()

def buildNmfData(calexps, footprint):
    """Build NMF data matrix
    
    Given an ordered dict of exposures in each band,
    create a matrix with rows as the image pixels in each band.
    
    Eventually we will also want to mask pixels, but for now we ignore masking.
    """
    # Since these are calexps, they should all have the same x0, y0 (initial pixel positions)
    x0 = calexps[0].getX0()
    y0 = calexps[0].getY0()
    
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
    variance = np.zeros((bandCount, pixelCount), dtype=np.int64)
    for n, calexp in enumerate(calexps):
        img, m, var = calexp.getMaskedImage().getArrays()
        data[n] = img[ymin:ymax, xmin:xmax].flatten()
        mask[n] = m[ymin:ymax, xmin:xmax].flatten()
        variance[n] = var[ymin:ymax, xmin:xmax].flatten()
    
    return data, mask, variance

def initNmfFactors(footprint, calexps):
    bbox = calexps[0].getBBox()
    xmin = bbox.getMinX()
    ymin = bbox.getMinY()
    peaks = []
    seds = np.zeros((len(calexps), len(footprint.getPeaks())))
    height, width = (footprint.getBBox().getHeight(), footprint.getBBox().getWidth())
    intensities = np.zeros((len(footprint.getPeaks()), width*height))
    for pk, peak in enumerate(footprint.getPeaks()):
        py, px = (peak.getIy()-ymin, peak.getIx()-xmin)
        for exp, calexp in enumerate(calexps):
            img = calexp.getMaskedImage().getImage().getArray()
            seds[exp, pk] = img[py, px]
    norm = np.sum(seds, axis=0)
    seds = seds/norm
    
    bbox = footprint.getBBox()
    xmin = bbox.getMinX()
    ymin = bbox.getMinY()
    for pk, peak in enumerate(footprint.getPeaks()):
        py, px = (peak.getIy()-ymin, peak.getIx()-xmin)
        intensities[pk, py*width + px] = norm[pk]
    return seds, intensities

def getPeakSymmetry(shape, px, py):
    """Build the operator to symmetrize a the intensities for a single row
    """
    center = (np.array(shape)-1)/2.0
    # If the peak is centered at the middle of the footprint,
    # make the entire footprint symmetric
    if px==center[1] and py==center[0]:
        return np.fliplr(np.eye(shape[0]*shape[1]))
    
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

def getPeakSymmetryOperator(shape, px, py):
    """Operator to calculate the difference from the symmetric intensities
    """
    symOp = getPeakSymmetry(shape, px, py)
    diffOp = scipy.sparse.identity(symOp.shape[0])-symOp
    return diffOp

def getSymmetryOperator(footprint):
    """Build the symmetry operator for an entire intensity matrix
    """
    symmetryOp = []
    bbox = footprint.getBBox()
    shape = (bbox.getHeight(),bbox.getWidth())
    for n,peak in enumerate(footprint.getPeaks()):
        px = peak.getIx()-footprint.getBBox().getMinX()
        py = peak.getIy()-footprint.getBBox().getMinY()
        s = getPeakSymmetryOperator(shape, px, py)
        symmetryOp.append(s)
    return symmetryOp

def getRadialMonotonicOp(shape, px, py):
    """Get a 2D operator to contrain radial monotonicity
    
    The monotonic operator basically calculates a radial the gradient in from the edges to the peak.
    Operating the monotonicity operator on a flattened image makes all non-monotonic pixels negative,
    which can then be projected to the subset gradient=0 using proximal operators.
    
    The radial monotonic operator is a sparse matrix, where each diagonal element is -1
    (except the peak position px, py) and each row has one other non-zero element, which is the
    closest pixel that aligns with a radial line from the pixel to the peak position.
    
    See DM-9143 for more.
    
    TODO: Implement this in C++ for speed
    """
    height, width = shape
    center = py*width+px
    monotonic = -np.eye(width*height, width*height)
    monotonic[center, center] = 0

    # Set the pixel in line with the radius to 1 for each pixel
    for h in range(height):
        for w in range(width):
            if h==py and w==px:
                continue
            dx = px-w
            dy = py-h
            pixel = h*width + w
            if px-w>py-h:
                if px-w>=h-py:
                    x = w + 1
                    y = h + int(np.round(dy/dx))
                elif px-w<h-py:
                    x = w - int(np.round(dx/dy))
                    y = h - 1
            elif px-w<py-h:
                if px-w>=h-py:
                    x = w + int(np.round(dx/dy))
                    y = h + 1
                elif px-w<h-py:
                    x = w - 1
                    y = h - int(np.round(dy/dx))
            else:
                if w<px:
                    x = w + 1
                    y = h + 1
                elif w>px:
                    x = w - 1
                    y = h - 1
            monotonic[pixel, y*width+x] = 1

    return scipy.sparse.coo_matrix(monotonic)

def getMonotonicOperator(footprint, getMonotonic=getRadialMonotonicOp):
    """Build the monotonicity operator for an entire intensity matrix
    
    Type can be "radial", "x", "y"
    """
    monotonicOp = []
    bbox = footprint.getBBox()
    shape = (bbox.getHeight(),bbox.getWidth())
    for n,peak in enumerate(footprint.getPeaks()):
        px = peak.getIx()-footprint.getBBox().getMinX()
        py = peak.getIy()-footprint.getBBox().getMinY()
        m = getMonotonic(shape, px, py)
        monotonicOp.append(m)
    return monotonicOp

def compareMeasToSim(footprint, seds, intensities, realTable, filters, vmin=None, vmax=None,
                     display=False):
    """Compare measurements to simulated "true" data
    
    If running nmf on simulations, this matches the detections to the simulation catalog and
    compares the measured flux of each object to the simulated flux.
    """
    peakCoords = np.array([[peak.getIx(),peak.getIy()] for peak in footprint.getPeaks()])
    refCoords = np.array(list(zip(realTable['x'], realTable['y'])))
    idx, d2 = debUtils.query_reference(peakCoords, refCoords, kdtree=None, 
            pool_size=None, separation=3, radec=False)
    shape = (footprint.getBBox().getHeight(), footprint.getBBox().getWidth())
    
    for k in range(len(seds[0])-1):
        logger.info("Object {0} at ({1},{2})".format(k, footprint.getPeaks()[k].getIx(),
                                                     footprint.getPeaks()[k].getIx()))
        for fidx, f in enumerate(filters):
            template = reconstructTemplate(seds, intensities, fidx , pkIdx=k, shape=shape)
            measFlux = np.sum(template)
            realFlux = realTable[idx][k]['flux_'+f]
            logger.info("Filter {0}: flux={1:.1f}, real={2:.1f}, error={3:.2f}%".format(
                f, measFlux, realFlux, 100*np.abs(measFlux-realFlux)/realFlux))
            if display:
                kwargs = {}
                if vmin is not None:
                    kwargs["vmin"] = vmin
                if vmax is not None:
                    kwargs["vmax"] = vmax*10
                plt.imshow(theory, interpolation='none', cmap='inferno', **kwargs)
                plt.show()
    return realTable[idx]

def noStepUpdate(stepsize, step, **kwargs):
    return stepsize

class DeblendedParent:
    def __init__(self, expDeblend, footprint, peaks):
        self.expDeblend = expDeblend
        self.filters = expDeblend.filters
        self.calexps = expDeblend.calexps
        self.psfs = expDeblend.psfs
        self.footprint = footprint
        self.peaks = peaks
        self.shape = (self.footprint.getBBox().getHeight(), self.footprint.getBBox().getWidth())
        
        # Initialize attributes to be assigned later
        self.data = None
        self.mask = None
        self.variance = None
        self.psfs = None
        self.initSeds = None
        self.initIntensities = None
        self.seds = None
        self.intensities = None
        self.psfOp = None
        self.symmetryOp = None
        self.monotonicOp = None

    def initNMF(self, initPsf=False, displaySeds=True, displayTemplates=True,
                      imgLimits=True, **displayKwargs):
        """Initialize the parameters needed for NMF deblending and (optionally) display the results
        """
        # Create the data matrices
        self.data, self.mask, self.variance = buildNmfData(self.calexps, self.footprint)
        self.initSeds, self.initIntensities = initNmfFactors(self.footprint, self.calexps)
    
        # Create the PSF Operator
        if initPsf:
            raise NotImplementedError("The PSF Operator is not yet implemented")
        else:
            self.psfOp = None
        
        if displaySeds:
            plotSeds(self.initSeds)
        if displayTemplates:
            if "fidx" not in displayKwargs:
                displayKwargs["fidx"] = 0
            if imgLimits:
                if "vmin" not in displayKwargs:
                    displayKwargs["vmin"] = self.expDeblend.vmin[displayKwargs["fidx"]]
                if "vmax" not in displayKwargs:
                    displayKwargs["vmax"] = 10*self.expDeblend.vmax[displayKwargs["fidx"]]
            plotIntensities(self.initSeds, self.initIntensities, self.shape, **displayKwargs)

        return self.data, self.mask, self.variance, self.initSeds, self.initIntensities
    
    def getSymmetryOp(self):
        """Create the operator to constrain symmetry
        """
        self.symmetryOp = getSymmetryOperator(self.footprint)
        return self.symmetryOp
    
    def getMonotonicOp(self, getMonotonic=getRadialMonotonicOp):
        """Create the operator to constrain monotonicity
        """
        self.monotonicOp = getMonotonicOperator(self.footprint, getMonotonic)
        return self.monotonicOp
    
    def deblend(self, constraints, displayKwargs=None, maxiter=1000, stepsize = 2, stepUpdate=noStepUpdate,
                display=False, imgLimits=True, **updateKwargs):
        """Run the NMF deblender
        
        This will always start from self.initW and self.initH, which can be modified before execution
        """
        if displayKwargs is None:
            displayKwargs = {}
        seds = np.copy(self.initSeds)
        intensities = np.copy(self.initIntensities)

        # Update W and H using the specified update function
        step = 0
        while step<maxiter:
            #newSeds = proxSed(newSeds, newIntensities, self.data, stepsize, self.psfOp, self.variance)
            #newIntensities = proxIntensities(newIntensities, newSeds, self.data, stepsize, self.psfOp,
            #                                 self.variance, constraints, constraintMatrices)
            step += 1
            stepsize = stepUpdate(stepsize, step)

        if display:
            # Show information about the fit
            for fidx, f in enumerate(self.filters):
                diff = (np.dot(seds, intensities)-self.data)[fidx].reshape(self.shape)
                logger.info('Filter {0}'.format(f))
                logger.info('Pixel range: {0} to {1}'.format(np.min(self.data), np.max(self.data)))
                logger.info('Max difference: {0}'.format(np.max(diff)))
                logger.info('Residual difference {0:.1f}%'.format(
                    100*np.abs(np.sum(diff)/np.sum(self.data[fidx]))))
            if self.expDeblend.simTable is not None:
                compareMeasToSim(self.footprint, seds, intensities, self.expDeblend.simTable, 
                                 self.filters, display=False)

            # Show the new templates for each object
            if "fidx" not in displayKwargs:
                displayKwargs["fidx"] = 0
            if imgLimits:
                if "vmin" not in displayKwargs:
                    displayKwargs["vmin"] = self.expDeblend.vmin[displayKwargs["fidx"]]
                if "vmax" not in displayKwargs:
                    displayKwargs["vmax"] = 10*self.expDeblend.vmax[displayKwargs["fidx"]]
            plotIntensities(seds, intensities, self.shape, **displayKwargs)
            plotSeds(seds)
            plt.imshow(diff, interpolation='none', cmap='inferno')
            plt.show()

        self.seds = seds
        self.intensities = intensities
        return seds, intensities

    def getTemplate(self, fidx, pkIdx, seds=None, intensities=None):
        if seds is None:
            seds = self.seds
        if intensities is None:
            intensities = self.intensities
        return reconstructTemplate(seds, intensities, fidx , pkIdx, self.shape)

    def displayTemplate(self, fidx, pkIdx, seds=None, intensities=None, imgLimits=True, 
                        cmap='inferno', **displayKwargs):
        template = self.getTemplate(fidx, pkIdx, seds, intensities)
        if imgLimits:
            if "vmin" not in displayKwargs:
                displayKwargs["vmin"] = self.expDeblend.vmin[self.filters[fidx]]
            if "vmax" not in displayKwargs:
                displayKwargs["vmax"] = 10*self.expDeblend.vmax[self.filters[fidx]]
        plt.imshow(template, interpolation='none', cmap=cmap, **displayKwargs)
        plt.show()


class ExposureDeblend:
    """Container for the objects and results of the NMF deblender
    """
    def __init__(self, filters, imgFilename, mergedDetFilename, simFilename=None):
        self.filters = filters

        # Initialize attributes to be assigned later
        self.calexps = None
        self.vmin = None
        self.vmax = None
        self.mergedDet = None
        self.mergedTable = None
        self.simCat = None
        self.simTable = None
        self.psfs = None
        self.deblends = None

        # Load Images and Catalogs
        self.loadFiles(imgFilename, mergedDetFilename, simFilename)
    
    def loadFiles(self, imgFilename=None, mergedDetFilename=None, simFilename=None):
        """Load images in each filter, the merged catalog and (optionally) a sim catalog
        """
        if imgFilename is not None:
            self.imgFilename = imgFilename
            self.calexps, self.vmin, self.vmax = loadCalExps(self.filters, imgFilename)
        if mergedDetFilename is not None:
            self.mergedDetFilename = mergedDetFilename
            self.mergedDet, self.mergedTable = loadMergedDetections(mergedDetFilename)
        if simFilename is not None:
            self.simFilename = simFilename
            self.simCat, self.simTable = sim.loadSimCatalog(simFilename)
        self.psfs = [calexp.getPsf() for calexp in self.calexps]
    
    def getParentFootprint(self, parentIdx=0, condition=None, display=True, imgLimits=True, **displayKwargs):
        """Get the parent footprint, peaks, and (optionally) display them
        """
        if condition is None:
            condition = np.ones((len(self.mergedTable),), dtype=bool)
        if display:
            if "fidx" not in displayKwargs:
                displayKwargs["fidx"] = 0
            if imgLimits:
                if "vmin" not in displayKwargs:
                    displayKwargs["vmin"] = self.vmin[displayKwargs["fidx"]]
                if "vmax" not in displayKwargs:
                    displayKwargs["vmax"] = 10*self.vmax[displayKwargs["fidx"]]
        # Load the footprint and peaks for parent[parentIdx]
        footprint, peaks = getParentFootprint(self.mergedTable, self.mergedDet, self.calexps,
                                              condition, parentIdx, display, **displayKwargs)
        return footprint, peaks
    
    def deblendParent(self, parentIdx=0, condition=None, initPsf=False, display=False,
                      displaySeds=False, displayTemplates=False, imgLimits=False,
                      constraints="m", **displayKwargs):
        footprint, peaks = self.getParentFootprint(parentIdx, condition, display, imgLimits, **displayKwargs)
        deblend = DeblendedParent(self, footprint, peaks)
        deblend.initNMF(initPsf, displaySeds, displayTemplates, imgLimits)
        # Only load the operators used in the constraints
        if "s" in constraints.lower():
            deblend.getSymmetryOp()
        if "m" in constraints.lower():
            deblend.getMonotonicOp()
        deblend.deblend(constraints=constraints, display=display)
        return deblend
    
    def deblend(self, condition=None, initPsf=False, constraints=""):
        self.deblendedParents = OrderedDict()
        for parentIdx, src in enumerate(self.mergedDet):
            if len(src.getFootprint().getPeaks())>1:
                result = self.deblendParent(parentIdx, condition, initPsf, constraints=constraints)
                self.deblendedParents[parentIdx] = result
    