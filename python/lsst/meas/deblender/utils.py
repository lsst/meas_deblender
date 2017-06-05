# Utilities useful for developing the deblender.
# This file is not expected to be merged to master, although subsets of it might be ported later
from collections import OrderedDict

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

def templateToFootprint(template, bbox, peak, thresh=1e-13, heavy=False):
    """Convert a template image into a Footprint

    There is currently no way in the stack to convert an image
    array into a spanset. The temporary workaround
    (recommended by jbosch) is to create a new
    FootprintSet by using a threshold, which will
    automatically create the new footprint.
    """
    import lsst.afw.image as afwImage
    import lsst.afw.detection as afwDet
    from lsst.afw.geom import SpanSet, Span

    # convert the numpy array to afw Image
    img = afwImage.ImageF(template)
    img = afwImage.MaskedImageF(img)
    # Shift the image to the correct coordinates
    img.setXY0(bbox.getMin())
    # Create the Footprint
    if thresh is None:
        # Use the entire footprint BBox as the footprint
        spans = SpanSet([Span(n, 0, bbox.getWidth()-1) for n in range(bbox.getHeight())])
        spans = spans.shiftedBy(bbox.getMinX(), bbox.getMinY())
        fp = afwDet.Footprint(spans)
    else:
        fps = afwDet.FootprintSet(img, afwDet.Threshold(thresh))
        # There should only be one footprint detected in the template
        if len(fps.getFootprints()) == 1:
            fp = fps.getFootprints()[0]
        elif len(fps.getFootprints()) > 1:
            spans = [fp.spans for fp in fps.getFootprints()]
            span = spans[0]
            for s in spans[1:]:
                span = span.union(s)
            fp = afwDet.Footprint(span)
        else: # no pixels above the threshold
            fp = afwDet.Footprint()

    # Clear the peak catalog detected by FootprintSet and add
    # the location of the peak
    fp.getPeaks().clear()
    fp.addPeak(peak[0], peak[1], template[int(peak[1])-bbox.getMinY(), int(peak[0])-bbox.getMinX()])
    if heavy:
        fp = afwDet.makeHeavyFootprint(fp, img)
    return fp

def makeMeasurementConfig(nsigma=6.0, nIterForRadius=1, kfac=2.5):
    """Construct a SingleFrameMeasurementConfig with the requested parameters"""
    import lsst.meas.extensions.photometryKron
    import lsst.meas.modelfit
    from lsst.meas.base import SingleFrameMeasurementConfig

    msConfig = SingleFrameMeasurementConfig()
    msConfig.algorithms.names = ["base_SdssCentroid",
                                 "base_SdssShape",
                                 "ext_photometryKron_KronFlux",
                                 "modelfit_DoubleShapeletPsfApprox",
                                 "modelfit_CModel"]
    msConfig.slots.centroid = "base_SdssCentroid"
    msConfig.slots.shape = "base_SdssShape"
    msConfig.slots.apFlux = "ext_photometryKron_KronFlux"
    msConfig.slots.modelFlux = "modelfit_CModel"
    msConfig.slots.psfFlux = None
    msConfig.slots.instFlux = None
    msConfig.slots.calibFlux = None
    # msConfig.algorithms.names.remove("correctfluxes")
    msConfig.plugins["ext_photometryKron_KronFlux"].nSigmaForRadius = nsigma
    msConfig.plugins["ext_photometryKron_KronFlux"].nIterForRadius = nIterForRadius
    msConfig.plugins["ext_photometryKron_KronFlux"].nRadiusForFlux = kfac
    msConfig.plugins["ext_photometryKron_KronFlux"].enforceMinimumRadius = False
    return msConfig

def makeExpMeasurements(fidx, calexps=None, templates=None, parent=None,
                        schema=None, config=None, thresh=1e-13,
                        deblenderResult=None, useEntireImage=False):
    """Make SingleFrameMeasurementTask measurements for a single exposure

    This requires either a list of `ExposureF`s, templates, and a parent
    (`Footprint`) to build footprints for each peak, or a `DeblenderResult`.

    Parameters
    ----------
    fidx: int
        Index of the filter to use. If using `ExposureF`s, this is the
        index of the calexp and template to use for the measurment.
    calexps: list of calexp's (`lsst.afw.image.imageLib.ExposureF`), default is None
        List of calibrated exposures to use for the measurment.
    templates: `numpy.ndarray`, default is None
        A 4 dimensional array (peak, band, y, x) that contains a
        model template for each peak.
    parent: `lsst.afw.detection.Footprint`, default is None
        The parent footprint containing the peaks in templates
    schema: `lsst.afw.table.Schema`, default is None
        Schema to use for the measurements. If no ``schema`` is specified
        then the minimal schema is used.
    config: config class, default is None
        Config class (containing the measurement plugins to use).
    thresh: float or None, default=1e-13
        Threshold for Footprint detection. Typically, with the templates created
        by the pipeline, this just has to be any number greater than zero.
        If ``thresh`` is ``None``, then entire bounding box for the
        ``parent`` `Footprint` is used.
    deblenderResult: `lsst.meas.deblender.baseline.DeblendedParent`, default is None
        Result from the old deblender. This is only required when using
        `Footprint`s from the old deblender.

    Returns
    -------
    sources: `lsst.afw.table.SourceCatalog`
    """
    from lsst.meas.algorithms.detection import SourceDetectionTask
    from lsst.meas.base import SingleFrameMeasurementTask
    import lsst.afw.table as afwTable

    # By default use a minimal Schema and base measurement plugins
    # And create the measurement task
    if schema is None:
        schema = afwTable.SourceTable.makeMinimalSchema()
    if config is None:
        config = makeMeasurementConfig()
    measureTask = SingleFrameMeasurementTask(schema, config=config)

    # Most of the time we use the templates to generate Footprints with all
    # pixels above the threshold
    if deblenderResult is None:
        if any([f is None for f in [templates, calexps]]):
            raise Exception("Required either 'deblenderResult' or 'templates', 'calexps'")
        peaks = parent.getPeaks()
    else:
        peaks = deblenderResult.peaks

    table = afwTable.SourceTable.make(schema)
    sources = afwTable.SourceCatalog(table)
    for pk, peak in enumerate(peaks):
        if deblenderResult is None:
            if not useEntireImage:
                bbox = parent.getBBox()
                fp = templateToFootprint(templates[pk][fidx].astype(np.float32), bbox,
                                         (peak.getIx(), peak.getIy()), heavy=True, thresh=thresh)
            else:
                bbox = calexps[0].getBBox()
                fp = templateToFootprint(templates[pk][fidx].astype(np.float32), bbox,
                                         (peak.getIx(), peak.getIy()), heavy=True, thresh=None)
        else:
            fp = deblenderResult.peaks[pk].deblendedPeaks[dbr.peaks[pk].filters[fidx]].getFluxPortion()
        child = sources.addNew()
        child.setFootprint(fp)

    # Make the measurements
    measureTask.run(sources, calexps[fidx])
    return sources
