# Temporary file to test using proximal operators in the NMF deblender
from __future__ import print_function, division
from collections import OrderedDict
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lsst.afw.display import rgb
from . import utils as debUtils

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

def compareSeds(tables, filters, ax=None, show=True, color_cycle=None):
    """Compare SEDs from using multiple detection methods

    This allows the user to compare the SEDs of a common set of peaks using multiple detection
    methods.

    Parameters
    ----------
    tables: list of `astropy.table.Table`s and `numpy.ndarray`s
        Either a peak table, with flux columns ``flux_{i}``, where ``{i}`` is the name of a filter
        in ``filters``, or an SED matrix, where each column is the SED for a different peak.
    filters: list of strings
        Names of the filters in the ``peakTable`` and ``simTable``.
    ax: `matplotlib.axes`, default = None
        Optional axes to plot the SEDs.
    show: bool, default = True
        Whether or not to show the plots or just update ``ax`` with the new plots
    color_cycle: list, default = None
        A list of colors to use for plotting the peaks. If ``color_cycle=None`` then a
        default color_cycle is used.

    Returns
    -------
    allSeds: list of `numpy.ndarray`
        Seds for each table/matrix in ``tables``.
    """
    # If the user didn't specify an axis, create a new figure
    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
    # Use a default color cycle so that peaks have consistent colors in all tables
    if color_cycle is None:
        color_cycle = [u'#4c72b0', u'#55a868', u'#c44e52', u'#8172b2', u'#ccb974', u'#64b5cd']
    allSeds = []
    markers = [".-", ".--", ".:", ".-."]
    midx = 0
    for n, tbl in enumerate(tables):
        # If tbl is an array, it is already an SED matrix
        if hasattr(tbl, "shape"):
            seds = tbl
        # otherwise extract the SED matrix from tbl
        else:
            seds = np.array([np.array(tbl["flux_{0}".format(f)]).tolist() for f in filters])
            norm = np.sum(seds, axis=0)
            seds = seds/norm
        # Plot the SED for each peak for the current table
        cidx = 0
        for pk in range(seds.shape[1]):
            if n==0:
                label = "Peak {0}".format(pk)
            else:
                label=None
            ax.plot(seds[:, pk], markers[midx], label=label, color=color_cycle[cidx])
            cidx += 1
            if cidx==len(color_cycle):
                cidx = 0
        allSeds.append(seds)
        midx += 1
        if midx>len(markers):
            midx = 0
    # Display the plot
    if show:
        plt.title("SEDs")
        plt.legend(loc="center left", fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(1, 0.5))
        plt.show()
    return allSeds

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

def imagesToRgb(images=None, calexps=None, filterIndices=None, xRange=None, yRange=None, **kwargs):
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
        filterIndices = [3,2,1]
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
    images = images.astype(np.float32)
    try:
        colors = rgb.AsinhZScaleMapping(images, **kwargs)
    except:
        logger.warning("Could not use ZScale, using  AsinhMapping with scaling")
        colors = rgb.AsinhMapping(np.min(images), np.max(images)-np.min(images))

    return colors.makeRgbImage(*images)

def plotColorImage(images=None, calexps=None, filterIndices=None, xRange=None, yRange=None,
                   Q=8, ax=None, show=True):
    """Display a collection of images or calexp's as an RGB image

    See `imagesToRgb` for more info.
    """
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
    # afw display struggles if one of the templates has no flux
    if images is not None:
        adjustedImages = images.copy()
        for i, img in enumerate(adjustedImages):
            if np.sum(img) == 0:
                adjustedImages[i][0][0] += 1e-9
    else:
        adjustedImages = images
    colors = imagesToRgb(adjustedImages, calexps, filterIndices, xRange, yRange, Q=Q)
    ax.imshow(colors)
    if show:
        plt.show()
    return ax

def maskPlot(img, mask=None, hideAxes=True, show=True, **kwargs):
    """Plot an image with specified pictures masked out

    It is often convenient to mask zero (or low flux) pixels in an image to highlight actual structure,
    so this convenience function makes it quick and easy to implement
    ``plt.plot(np.ma.array(img, mask=mask), **kwargs)``, optionally hiding the axes and showing the image.
    """
    if mask is None:
        mask = img==0
    maImg = np.ma.array(img, mask=mask)

    plt.imshow(maImg, **kwargs)
    if hideAxes:
        plt.axis("off")
    if show:
        plt.show()
    return plt

def plotImgWithMarkers(calexps, footprint, filterIndices=None, show=True,
                       ax=None, img_kwargs=None, footprint_kwargs=None, Q=8, **plot_kwargs):
    """Plot an RGB image with the footprint and peaks marked

    Use the bounding box of a footprint to extract image data from a set of calexps in a set of colors
    and plot the image, with the outline of the footprint and the footprint peaks marked
    """
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
    if img_kwargs is None:
        img_kwargs = {}
    if footprint_kwargs is None:
        footprint_kwargs = {}
    bbox = footprint.getBBox()
    refBbox = calexps[0].getMaskedImage().getBBox()

    # Display the full color image
    xSlice, ySlice = debUtils.getRelativeSlices(bbox, refBbox)
    colors = imagesToRgb(calexps=calexps, filterIndices=filterIndices, xRange=xSlice, yRange=ySlice,
                         Q=Q)
    ax.imshow(colors, **img_kwargs)

    # Display the footprint border
    border, filled = debUtils.getFootprintArray(footprint)
    if "interpolation" not in footprint_kwargs:
        footprint_kwargs["interpolation"] = "none"
    if "cmap" not in footprint_kwargs:
        footprint_kwargs["cmap"] = "cool"
    ax.imshow(border, **footprint_kwargs)

    px = [peak.getIx()-bbox.getMinX() for peak in footprint.getPeaks()]
    py = [peak.getIy()-bbox.getMinY() for peak in footprint.getPeaks()]
    ax.plot(px, py, "cx", mew=2, **plot_kwargs)
    ax.set_xlim(0,colors.shape[1]-1)
    ax.set_ylim(colors.shape[0]-1, 0)
    if show:
        plt.show()
    return ax

def plotFluxDifference(tables, simTable, filters, ax=None, show=True, color_cycle=None):
    """Plot the difference between measurements and simulated data

    Given a set of peakTables, compare the flux in each band to
    simultated data.

    Parameters
    ----------
    tables: list of `astropy.table.Table`
        Either a peak table, with flux columns ``flux_{i}``, where ``{i}`` is the name of a filter
        in ``filters``, or an SED matrix, where each column is the SED for a different peak.
    simTable: `astropy.table.Table`
        A table that has been matched with a `peakTable`
    filters: list of strings
        Names of the filters in the ``peakTable`` and ``simTable``.
    ax: `matplotlib.axes`, default = None
        Optional axes to plot the SEDs.
    show: bool, default = True
        Whether or not to show the plots or just update ``ax`` with the new plots
    color_cycle: list, default = None
        A list of colors to use for plotting the peaks. If ``color_cycle=None`` then a
        default color_cycle is used.

    Returns
    -------
    None
    """
    # If the user didn't specify an axis, create a new figure
    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
    # Use a default color cycle so that peaks have consistent colors in all tables
    if color_cycle is None:
        color_cycle = [u'#4c72b0', u'#55a868', u'#c44e52', u'#8172b2', u'#ccb974', u'#64b5cd']
    markers = [".-", ".--", ".:", ".-."]
    midx = 0
    for n, tbl in enumerate(tables):
        cidx = 0
        for pk in range(len(simTable)):
            if n==0:
                label = "Peak {0}".format(pk)
            else:
                label = None
            flux_diff = []
            for f in filters:
                diff = ((tbl["flux_"+f][pk]-simTable["flux_"+f][pk])/simTable["flux_"+f][pk])
                flux_diff.append(diff)
            ax.plot(flux_diff, markers[midx], color=color_cycle[cidx], label=label)
            cidx += 1
            if cidx==len(color_cycle):
                cidx = 0

        midx += 1
        if midx==len(markers):
            midx = 0

    if show:
        ax.set_xlabel("Peak Number")
        ax.set_ylabel("(Measured-Sim)/Sim Flux")
        plt.legend(loc="center left", fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(1, 0.5))
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        plt.show()

def plotPeakTemplates(templates, columns=3, figsize=None, **plotKwargs):
    """Plot a set of templates for a given peak

    Parameters
    ----------
    templates: dict
        Dictionary of templates for the given peak.
        The keys of the dictionary will be the title for each image
        while the values are the templates themselves.
    columns: int, default=3
        Number of columns in the figure
    figsize: tuple, None
        Size of the figure. If None, this will be calculated automatically,
        using a width of size 12 and a height proportional to the shape of
        the images.
    plotKwargs:
        Optional keyword arguments passed to
        `lsst.meas.deblender.display.plotColorImage`.

    Returns
    -------
    None
    """
    titles = list(templates.keys())
    shape = templates[titles[0]][0].shape
    ratio = shape[0]/shape[1]
    # Calculate the figure grid and create the figure
    rows = 1+len(templates)//columns
    if np.mod(len(templates),columns)==0:
        rows -= 1
    fig = plt.figure(figsize=(12, 12*ratio/1.618))
    # Plot the image using all of the templates
    for n, (title, template) in enumerate(templates.items()):
        ax = fig.add_subplot(rows, columns, n+1)
        ax.axis("off")
        plotColorImage(template, ax=ax, show=False, **plotKwargs)
        ax.set_title(title)
    plt.show()

def plotAllTemplates(allTemplates, columns=3, figsize=None, **plotKwargs):
    """Plot a set of templates for all peaks in a blend

    Parameters
    ----------
    allTemplates: dict
        Dictionary of templates.
        The keys of the dictionary will be the title for each image
        while the values are 3D templates, with axes (peak, y, x).

    See `plotPeakTemplates` for a description of the other parameters
    """
    _, template = list(allTemplates.items())[0]
    for pk in range(len(template)):
        logger.info("Peak {0}".format(pk))
        templates = OrderedDict([(t, template[pk]) for t, template in allTemplates.items()])
        plotPeakTemplates(templates, columns=columns, figsize=figsize, **plotKwargs)
