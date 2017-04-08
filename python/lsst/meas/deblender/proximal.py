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
from . import proximal_nmf as pnmf
from . import display as debDisplay

logging.basicConfig()
logger = logging.getLogger("lsst.meas.deblender.proximal")

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

def getParentFootprint(mergedTable, mergedDet, calexps, condition, parentIdx, display=True,
                       filterIndices=None, contrast=100, adjustZero=True):
    """Load the parent footprint and peaks, and (optionally) display the image and footprint border
    """
    idx = np.where(condition)[0][parentIdx]
    src = mergedDet[idx]
    fp = src.getFootprint()
    peaks = fp.getPeaks()
    
    if display:
        bbox = fp.getBBox()
        refBbox = calexps[0].getMaskedImage().getBBox()
        
        # Display the full color image
        xSlice, ySlice = debUtils.getRelativeSlices(bbox, refBbox)
        colors = debDisplay.imagesToRgb(calexps=calexps, filterIndices=filterIndices,
                                        xRange=xSlice, yRange=ySlice,
                                        contrast=contrast, adjustZero=adjustZero)
        plt.imshow(colors)
        
        # Display the footprint border
        border, filled = debUtils.getFootprintArray(src)
        plt.imshow(border, interpolation='none', cmap='cool')

        px = [peak.getIx()-bbox.getMinX() for peak in fp.getPeaks()]
        py = [peak.getIy()-bbox.getMinY() for peak in fp.getPeaks()]
        plt.plot(px, py, "cx")
        plt.xlim(0,colors.shape[1]-1)
        plt.ylim(colors.shape[0]-1, 0)
        plt.show()
    return fp, peaks

def reconstructTemplate(seds, intensities, fidx , pkIdx, shape=None, psfOp=None):
    """Calculate the template for a single peak for a single filter
    
    Use the SED matrix ``seds``, intensity matrix ``intensities``, and optionally the PSF operator,
     to reconstruct the flux in the  image due to the selected peak, in the selected filter.
    """
    # Optionally convolve the intensities with the PSF
    if psfOp is not None:
        img = psfOp[fidx].dot(intensities[pkIdx,:].flatten()).reshape(intensities[pkIdx,:].shape)
    else:
        img = intensities[pkIdx]
    template = seds[fidx,pkIdx]*img
    if shape is not None:
        template = template.reshape(shape)
    return template

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
    
    # Add the image in each filter as a row in data
    data = np.zeros((bandCount, bbox.getHeight(), bbox.getWidth()), dtype=np.float64)
    mask = np.zeros((bandCount, bbox.getHeight(), bbox.getWidth()), dtype=np.int64)
    variance = np.zeros((bandCount, bbox.getHeight(), bbox.getWidth()), dtype=np.float64)
    for n, calexp in enumerate(calexps):
        img, m, var = calexp.getMaskedImage().getArrays()
        data[n] = img[ymin:ymax, xmin:xmax]
        mask[n] = m[ymin:ymax, xmin:xmax]
        variance[n] = var[ymin:ymax, xmin:xmax]
    
    return data, mask, variance

def compareMeasToSim(footprint, seds, intensities, realTable, filters, vmin=None, vmax=None,
                     display=False, poolSize=-1, psfOp=None, fluxPortions=None):
    """Compare measurements to simulated "true" data
    
    If running nmf on simulations, this matches the detections to the simulation catalog and
    compares the measured flux of each object to the simulated flux.
    """
    peakCoords = np.array([[peak.getIx(),peak.getIy()] for peak in footprint.getPeaks()])
    simCoords = np.array(list(zip(realTable['x'], realTable['y'])))
    kdtree = scipy.spatial.cKDTree(simCoords)
    d2, idx = kdtree.query(peakCoords, n_jobs=poolSize)
    shape = (footprint.getBBox().getHeight(), footprint.getBBox().getWidth())
    
    for k in range(len(seds[0])):
        logger.info("Object {0} at ({1},{2})".format(k, footprint.getPeaks()[k].getIx(),
                                                     footprint.getPeaks()[k].getIy()))
        for fidx, f in enumerate(filters):
            template = reconstructTemplate(seds, intensities, fidx , pkIdx=k, shape=shape, psfOp=psfOp)
            measFlux = np.sum(template)
            realFlux = realTable[idx][k]['flux_'+f]
            logger.info("Filter {0}: template flux={1:.1f}, real={2:.1f}, error={3:.2f}%".format(
                        f, measFlux, realFlux, 100*np.abs(measFlux-realFlux)/realFlux))
        for fidx, f in enumerate(filters):
            template = reconstructTemplate(seds, intensities, fidx , pkIdx=k, shape=shape, psfOp=psfOp)
            realFlux = realTable[idx][k]['flux_'+f]
            if fluxPortions is not None:
                flux = fluxPortions[fidx, k]
                logger.info("Filter {0}: re-apportioned flux={1:.1f}, real={2:.1f}, error={3:.2f}%".format(
                        f, flux, realFlux, 100*np.abs(flux-realFlux)/realFlux))
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
    def __init__(self, expDeblend, footprint, peaks, usePsf=False):
        self.expDeblend = expDeblend
        self.filters = expDeblend.filters
        self.calexps = expDeblend.calexps
        self.psfs = expDeblend.psfs
        self.psfThresh = expDeblend.psfThresh
        self.footprint = footprint
        self.bbox = footprint.getBBox()
        self.peaks = peaks
        self.shape = (self.bbox.getHeight(), self.bbox.getWidth())
        self.usePsf = usePsf
        
        # Get the position of the peaks
        # (needed by Peters code to calculate the initial matrices)
        x0 = self.calexps[0].getX0()
        y0 = self.calexps[0].getY0()
        xmin = self.bbox.getMinX()
        ymin = self.bbox.getMinY()
        peaks = [(pk.getIx()-xmin, pk.getIy()-ymin) for pk in self.peaks]
        self.peakCoords = peaks
        
        # Initialize attributes to be assigned later
        self.data = None
        self.mask = None
        self.variance = None
        self.initSeds = None
        self.initIntensities = None
        self.seds = None
        self.intensities = None
        self.symmetryOp = None
        self.monotonicOp = None
        self.psfOp = None
        self.cutoff = 0
        self.peakFlux = None
        self.correlations = None

    def initNMF(self, displaySeds=False, displayTemplates=False,
                filterIndices=None, contrast=100, adjustZero=True, usePsf=None):
        """Initialize the parameters needed for NMF deblending and (optionally) display the results
        """
        from . import proximal_nmf
        
        if usePsf is not None:
            self.usePsf = usePsf
        else:
            usePsf = self.usePsf
        
        # Create the data matrices
        self.data, self.mask, self.variance = buildNmfData(self.calexps, self.footprint)

        if displaySeds:
            debDisplay.plotSeds(self.initSeds)
        if displayTemplates:
            # The following steps are currently done in Peters deblender by we
            # initialize them here for vizualization
            B,N,M = self.data.shape
            K = len(self.peakCoords)
            self.initSeds = proximal_nmf.init_A(B, K, peaks=self.peakCoords, I=self.data)
            self.initIntensities = proximal_nmf.init_S(N, M, K, peaks=self.peakCoords, I=self.data)
            self.psfOp = [proximal_nmf.getPSFOp(psfImg, self.data[0].shape, threshold=self.psfThresh)
                            for psfImg in self.psfs]
            if filterIndices is None:
                filterIndices = [2,1,0]
            for pk in range(K):
                templates = [self.getTemplate(n, pk, self.initSeds, self.initIntensities) for n in filterIndices]
                templates = np.array(templates)
                debDisplay.plotColorImage(templates, filterIndices=range(len(templates)), contrast=contrast,
                                          adjustZero=adjustZero)
            #debDisplay.plotIntensities(self.initSeds, self.initIntensities, self.shape, **displayKwargs)

        return self.data, self.mask, self.variance, self.initSeds, self.initIntensities
    
    def getSymmetryOp(self):
        """Create the operator to constrain symmetry
        
        Currently this is implemented in Peters algorithm but it is likely to be moved to this class later
        """
        #self.symmetryOp = getSymmetryOperator(self.footprint)
        return #self.symmetryOp
    
    def getMonotonicOp(self):
        """Create the operator to constrain monotonicity
        
        Currently this is implemented in Peters algorithm but it is likely to be moved to this class later
        """
        #self.monotonicOp = getMonotonicOperator(self.footprint, getMonotonic)
        return #self.monotonicOp
    
    def deblend(self, constraints="M", displayKwargs=None, maxiter=50, stepsize = 2,
                display=False, filterIndices=None, contrast=100, adjustZero=False,
                psfThresh=None, usePsf=None, **kwargs):
        """Run the NMF deblender

        This currently just initializes the data (if necessary) and calls the nmf_deblender from
        proximal_nmf. It can also display the deblended footprints and statistics describing the
        fit if ``display=True``.
        """
        if displayKwargs is None:
            displayKwargs = {}

        if self.data is None:
            self.initNMF()

        # Set the variance for bad pixels to zero
        maskPlane = self.calexps[0].getMaskedImage().getMask().getMaskPlaneDict().asdict()
        badPixels = (1<<maskPlane["BAD"] |
                     1<<maskPlane["CR"] |
                     1<<maskPlane["NO_DATA"] |
                     1<<maskPlane["SAT"] |
                     1<<maskPlane["SUSPECT"])
        # Set the variance for pixels outside the footprint to zero
        import lsst.afw.image as afwImage
        import lsst.afw.detection as afwDetect
        fpMask = afwImage.MaskU(self.footprint.getBBox())
        afwDetect.setMaskFromFootprint(fpMask, self.footprint, 1)
        fpMask = ~fpMask.getArray().astype(bool)
        
        mask = ((badPixels & self.mask) | fpMask).astype(bool)
        variance = np.copy(self.variance)
        variance[mask] = 0
        data = self.data.copy()
        data[mask] = 0
        debDisplay.maskPlot(data[0], data[0]==0)
        

        if usePsf is None:
            usePsf = self.usePsf
        else:
            self.usePsf = usePsf
        if psfThresh is not None:
            self.psfThresh = psfThresh
        if usePsf and 'P' not in kwargs:
            #self.psfOp = [pnmf.getPSFOp(psfImg, self.data[0].shape, threshold=self.psfThresh)
            #              for psfImg in self.psfs]
            kwargs['P'] = self.psfs
        logger.info("constraints: {0}".format(constraints))
        result = pnmf.nmf_deblender(data, K=len(self.peakCoords), max_iter=maxiter,
                                    peaks=self.peakCoords, W=variance, constraints=constraints,
                                    psf_thresh=self.psfThresh, **kwargs)
        seds, intensities, self.model, self.psfOp, self.errors = result
        self.seds = seds
        self.intensities = intensities

        if display:
            bands = intensities.shape[0]
            peakCount = seds.shape[1]
            pixels = intensities.shape[1]*intensities.shape[2]
            # Show information about the fit
            for fidx, f in enumerate(self.filters):
                if self.usePsf:
                    convolved = [self.psfOp[fidx].dot(intensities[pk,:].flatten()).reshape(self.shape) for pk in range(peakCount)]
                    convolved = np.array(convolved)
                    model = np.dot(seds, convolved.reshape(bands, pixels)).reshape(self.data.shape)
                else:
                    model = np.dot(seds, intensities.reshape(bands, pixels)).reshape(self.data.shape)
                diff = (model-self.data)[fidx].reshape(self.shape)
                logger.info('Filter {0}'.format(f))
                logger.info('Pixel range: {0} to {1}'.format(np.min(self.data), np.max(self.data)))
                logger.info('Max difference: {0}'.format(np.max(diff)))
                logger.info('Residual difference {0:.1f}%'.format(
                    100*np.abs(np.sum(diff)/np.sum(self.data[fidx]))))
            if self.expDeblend.simTable is not None:
                compareMeasToSim(self.footprint, seds, intensities, self.expDeblend.simTable, 
                                 self.filters, display=False, psfOp=self.psfOp,
                                 fluxPortions=self.getFluxPortion())

            # Show the new templates for each object
            for pk in range(len(intensities)):
                templates = np.array([self.getTemplate(n, pk) for n in range(len(self.calexps))])
                debDisplay.plotColorImage(images=templates, filterIndices=filterIndices,
                                          contrast=contrast, adjustZero=adjustZero, figsize=(5,5))
            debDisplay.plotSeds(seds)
            plt.imshow(diff, interpolation='none', cmap='inferno')
            plt.show()

        return seds, intensities

    def getTemplate(self, fidx, pkIdx, seds=None, intensities=None):
        """Apply the SED to the intensities to get the template for a given filter
        """
        if seds is None:
            seds = self.seds
        if intensities is None:
            intensities = self.intensities
        if self.usePsf:
            psfOp = self.psfOp
        else:
            psfOp = None
        return reconstructTemplate(seds, intensities, fidx , pkIdx, self.shape, psfOp)

    def displayImage(self, pkIdx, fidx=0, imgType='template', useMask=True, cutoff=None, seds=None,
                        intensities=None, imgLimits=False, cmap='inferno', **displayKwargs):
        """Display an appropriately scaled template
        """
        if imgType.lower() == 'template':
            img = self.getTemplate(fidx, pkIdx, seds, intensities)
        elif imgType.lower() == 'intensity':
            img = self.intensities[pkIdx]
        else:
            raise Exception("imgType must be either 'template' or 'intensity'")
        if imgLimits:
            if "vmin" not in displayKwargs:
                displayKwargs["vmin"] = self.expDeblend.vmin[fidx]
            if "vmax" not in displayKwargs:
                displayKwargs["vmax"] = 10*self.expDeblend.vmax[fidx]
        if useMask:
            if cutoff is None:
                cutoff = self.cutoff
            else:
                cutoff = cutoff
            img = np.ma.array(img, mask=img<=cutoff)
        plt.imshow(img, interpolation='none', cmap=cmap, **displayKwargs)
        plt.show()

    def displayAllImages(self, fidx=0, imgType='template', useMask=True, cutoff=None,
                            imgLimits=False, cmap='inferno', **displayKwargs):
        for pk in range(len(self.intensities)):
            self.displayImage(pk, fidx, imgType, useMask=useMask, cutoff=cutoff,
                                 imgLimits=imgLimits, cmap=cmap, **displayKwargs)

    def trimFlux(self, cutoff=1e-2):
        seds = np.max(self.seds, axis=0)
        intensities = (self.intensities.T*seds).T
        self.intensities[intensities<cutoff] = 0
    
    def getFluxPortion(self):
        """Use the intensity templates to estimate the flux for all of the sources
        """
        filters = len(self.data)
        peakCount = len(self.intensities)
        peakFlux = np.zeros((filters, peakCount))
        for fidx in range(filters):
            data = self.data[fidx]
            weight = self.variance[fidx]
            totalWeight = np.sum(weight)
            if self.psfOp is not None:
                intensities = np.zeros_like(self.intensities)
                psfOp = self.psfOp[fidx]
                for pk in range(peakCount):
                    intensities[pk,:] = psfOp.dot(self.intensities[pk,:].flatten()).reshape(
                                                  self.intensities[pk,:].shape)
            else:
                intensities = self.intensities
            totalFlux = np.sum(intensities, axis=0)
            normalization = totalFlux*totalWeight
            zeroFlux = normalization==0
            normalization[zeroFlux] = 1
            normalization = 1/normalization

            for pk in range(peakCount):
                flux = weight*data*intensities[pk]*normalization
                flux[zeroFlux] = 0
                peakFlux[fidx][pk] = np.sum(flux)*data.size

        self.peakFlux = peakFlux
        return peakFlux

    def getCorrelations(self, minFlux=None):
        """Calculate the correlation between two footprints
        
        LSST Footprints will not be added to the NMF deblender until after the new footprints
        are merged with master. Until then, users can specify ``minFlux``, which will clip the
        "footprint" to regions with flux above ``minFlux``.
        """
        intensities = np.copy(self.intensities)
        if minFlux is not None:
            intensities[intensities<minFlux] = 0
        totalIntensity = np.sum(intensities, axis=(1,2))
        
        peakCount = len(self.peaks)
        correlations = np.zeros((peakCount, peakCount))
        for i in range(peakCount-1):
            for j in range(i+1, peakCount):
                norm = totalIntensity[i]*totalIntensity[j]
                if norm>0:
                    correlations[i,j] = np.sum(intensities[i]*intensities[j])/norm
        self.correlations = correlations
        return correlations
        
        templates = [self.getTemplate(0, pk) for pk in range(len(self.peaks))]
        if minFlux is not None:
            for n, t in enumerate(templates):
                templates[n][templates[n]<minFlux] = 0
        peakCount = len(self.peaks)
        degenerateFlux = np.zeros((peakCount, peakCount))
        for i in range(peakCount-1):
            for j in range(i+1, peakCount):
                norm = np.sum(templates[i])*np.sum(templates[j])
                if norm>0:
                    degenerateFlux[i,j] = np.sum(templates[i]*templates[j])/norm
        
        self.correlations = degenerateFlux
        return degenerateFlux

class ExposureDeblend:
    """Container for the objects and results of the NMF deblender
    """
    def __init__(self, filters, imgFilename, mergedDetFilename, simFilename=None, psfThresh=1e-2):
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
        self.psfThresh = psfThresh

        # Load Images and Catalogs
        self.loadFiles(imgFilename, mergedDetFilename, simFilename)
    
    def loadFiles(self, imgFilename=None, mergedDetFilename=None, simFilename=None):
        """Load images in each filter, the merged catalog and (optionally) a sim catalog
        """
        from . import proximal_nmf
        if imgFilename is not None:
            self.imgFilename = imgFilename
            self.calexps, self.vmin, self.vmax = loadCalExps(self.filters, imgFilename)
        if mergedDetFilename is not None:
            self.mergedDetFilename = mergedDetFilename
            self.mergedDet, self.mergedTable = loadMergedDetections(mergedDetFilename)
        if simFilename is not None:
            self.simFilename = simFilename
            self.simCat, self.simTable = sim.loadSimCatalog(simFilename)
        expPsfs = [calexp.getPsf() for calexp in self.calexps]
        # Normalize the PSF from each image to 1 (makes it more straight forward to set a cutoff and
        # initialize the psf operator)
        #self.psfs = [psf.computeImage().getArray()/np.max(psf.computeImage().getArray()) for psf in expPsfs]
        self.psfs = [psf.computeImage().getArray() for psf in expPsfs]
    
    def getParentFootprint(self, parentIdx=0, condition=None, display=True,
                           filterIndices=None, contrast=100, adjustZero=True):
        """Get the parent footprint, peaks, and (optionally) display them
        
        ``parentIdx`` is the index of the parent footprint in ``self.mergedTable[condition]``, where
        condition is some array or index used to select some subset of the catalog, for example
        ``self.mergedTable["peaks"]>0``.
        """
        if condition is None:
            condition = np.ones((len(self.mergedTable),), dtype=bool)
        # Load the footprint and peaks for parent[parentIdx]
        footprint, peaks = getParentFootprint(self.mergedTable, self.mergedDet, self.calexps,
                                              condition, parentIdx, display, filterIndices, contrast,
                                              adjustZero)
        return footprint, peaks
    
    def deblendParent(self, parentIdx=0, condition=None, initPsf=False, display=False,
                      displaySeds=False, displayTemplates=False, constraints="M", maxiter=50,
                      filterIndices=None, contrast=100, adjustZero=False, **kwargs):
        """Deblend a single parent footprint

        Deblend a parent selected by passing a ``parentIdx`` and ``condition``
        (see `ExposureDeblend.getParentFootprint`) and choosing a constraint
        ("M" for monotonicity, "S" for symmetry, and " " for no constraint) and
        maximum number of iterations (maxiter) for each step in the ADMM algorithm.
        """
        footprint, peaks = self.getParentFootprint(parentIdx, condition, display,
                                                   filterIndices, contrast, adjustZero)
        deblend = DeblendedParent(self, footprint, peaks)
        deblend.initNMF(initPsf, displaySeds, displayTemplates, filterIndices, contrast, adjustZero)
        deblend.deblend(constraints=constraints, maxiter=maxiter, display=display, **kwargs)
        return deblend
    
    def deblend(self, condition=None, initPsf=False, constraints="M", maxiter=50, **kwargs):
        """Deblend all of the footprints with multiple peaks
        """
        self.deblendedParents = OrderedDict()
        for parentIdx, src in enumerate(self.mergedDet):
            if len(src.getFootprint().getPeaks())>1:
                result = self.deblendParent(parentIdx, condition, initPsf,
                                            constraints=constraints, maxiter=maxiter, **kwargs)
                self.deblendedParents[src.getId()] = result
    