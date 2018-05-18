#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import math
import numpy as np
import time

import lsst.log
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable

logger = lsst.log.Log.getLogger("meas.deblender.deblend")

__all__ = 'SourceDeblendConfig', 'SourceDeblendTask', 'MultibandDeblendConfig', 'MultibandDeblendTask'


class SourceDeblendConfig(pexConfig.Config):

    edgeHandling = pexConfig.ChoiceField(
        doc='What to do when a peak to be deblended is close to the edge of the image',
        dtype=str, default='ramp',
        allowed={
            'clip': 'Clip the template at the edge AND the mirror of the edge.',
            'ramp': 'Ramp down flux at the image edge by the PSF',
            'noclip': 'Ignore the edge when building the symmetric template.',
        }
    )

    strayFluxToPointSources = pexConfig.ChoiceField(
        doc='When the deblender should attribute stray flux to point sources',
        dtype=str, default='necessary',
        allowed={
            'necessary': 'When there is not an extended object in the footprint',
            'always': 'Always',
            'never': ('Never; stray flux will not be attributed to any deblended child '
                      'if the deblender thinks all peaks look like point sources'),
        }
    )

    assignStrayFlux = pexConfig.Field(dtype=bool, default=True,
                                    doc='Assign stray flux (not claimed by any child in the deblender) '
                                        'to deblend children.')

    strayFluxRule = pexConfig.ChoiceField(
        doc='How to split flux among peaks',
        dtype=str, default='trim',
        allowed={
            'r-to-peak': '~ 1/(1+R^2) to the peak',
            'r-to-footprint': ('~ 1/(1+R^2) to the closest pixel in the footprint.  '
                               'CAUTION: this can be computationally expensive on large footprints!'),
            'nearest-footprint': ('Assign 100% to the nearest footprint (using L-1 norm aka '
                                  'Manhattan distance)'),
            'trim': ('Shrink the parent footprint to pixels that are not assigned to children')
        }
    )

    clipStrayFluxFraction = pexConfig.Field(dtype=float, default=0.001,
                                          doc=('When splitting stray flux, clip fractions below '
                                               'this value to zero.'))
    psfChisq1 = pexConfig.Field(dtype=float, default=1.5, optional=False,
                              doc=('Chi-squared per DOF cut for deciding a source is '
                                   'a PSF during deblending (un-shifted PSF model)'))
    psfChisq2 = pexConfig.Field(dtype=float, default=1.5, optional=False,
                              doc=('Chi-squared per DOF cut for deciding a source is '
                                   'PSF during deblending (shifted PSF model)'))
    psfChisq2b = pexConfig.Field(dtype=float, default=1.5, optional=False,
                               doc=('Chi-squared per DOF cut for deciding a source is '
                                    'a PSF during deblending (shifted PSF model #2)'))
    maxNumberOfPeaks = pexConfig.Field(dtype=int, default=0,
                                     doc=("Only deblend the brightest maxNumberOfPeaks peaks in the parent"
                                          " (<= 0: unlimited)"))
    maxFootprintArea = pexConfig.Field(dtype=int, default=1000000,
                                     doc=("Maximum area for footprints before they are ignored as large; "
                                          "non-positive means no threshold applied"))
    maxFootprintSize = pexConfig.Field(dtype=int, default=0,
                                     doc=("Maximum linear dimension for footprints before they are ignored "
                                          "as large; non-positive means no threshold applied"))
    minFootprintAxisRatio = pexConfig.Field(dtype=float, default=0.0,
                                          doc=("Minimum axis ratio for footprints before they are ignored "
                                               "as large; non-positive means no threshold applied"))
    notDeblendedMask = pexConfig.Field(dtype=str, default="NOT_DEBLENDED", optional=True,
                                     doc="Mask name for footprints not deblended, or None")

    tinyFootprintSize = pexConfig.RangeField(dtype=int, default=2, min=2, inclusiveMin=True,
                                           doc=('Footprints smaller in width or height than this value will '
                                                'be ignored; minimum of 2 due to PSF gradient calculation.'))

    propagateAllPeaks = pexConfig.Field(dtype=bool, default=False,
                                      doc=('Guarantee that all peaks produce a child source.'))
    catchFailures = pexConfig.Field(dtype=bool, default=False,
                                  doc=("If True, catch exceptions thrown by the deblender, log them, "
                                       "and set a flag on the parent, instead of letting them propagate up"))
    maskPlanes = pexConfig.ListField(dtype=str, default=["SAT", "INTRP", "NO_DATA"],
                                   doc="Mask planes to ignore when performing statistics")
    maskLimits = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={},
        doc=("Mask planes with the corresponding limit on the fraction of masked pixels. "
             "Sources violating this limit will not be deblended."),
    )
    weightTemplates = pexConfig.Field(dtype=bool, default=False,
                                    doc=("If true, a least-squares fit of the templates will be done to the "
                                        "full image. The templates will be re-weighted based on this fit."))
    removeDegenerateTemplates = pexConfig.Field(dtype=bool, default=False,
                                              doc=("Try to remove similar templates?"))
    maxTempDotProd = pexConfig.Field(dtype=float, default=0.5,
                                   doc=("If the dot product between two templates is larger than this value"
                                        ", we consider them to be describing the same object (i.e. they are "
                                        "degenerate).  If one of the objects has been labeled as a PSF it "
                                        "will be removed, otherwise the template with the lowest value will "
                                        "be removed."))
    medianSmoothTemplate = pexConfig.Field(dtype=bool, default=True,
                                         doc="Apply a smoothing filter to all of the template images")

## \addtogroup LSST_task_documentation
## \{
## \page SourceDeblendTask
## \ref SourceDeblendTask_ "SourceDeblendTask"
## \copybrief SourceDeblendTask
## \}


class SourceDeblendTask(pipeBase.Task):
    """!
    \anchor SourceDeblendTask_

    \brief Split blended sources into individual sources.

    This task has no return value; it only modifies the SourceCatalog in-place.
    """
    ConfigClass = SourceDeblendConfig
    _DefaultName = "sourceDeblend"

    def __init__(self, schema, peakSchema=None, **kwargs):
        """!
        Create the task, adding necessary fields to the given schema.

        @param[in,out] schema        Schema object for measurement fields; will be modified in-place.
        @param[in]     peakSchema    Schema of Footprint Peaks that will be passed to the deblender.
                                     Any fields beyond the PeakTable minimal schema will be transferred
                                     to the main source Schema.  If None, no fields will be transferred
                                     from the Peaks.
        @param[in]     **kwargs      Passed to Task.__init__.
        """
        pipeBase.Task.__init__(self, **kwargs)
        self.schema = schema
        peakMinimalSchema = afwDet.PeakTable.makeMinimalSchema()
        if peakSchema is None:
            # In this case, the peakSchemaMapper will transfer nothing, but we'll still have one
            # to simplify downstream code
            self.peakSchemaMapper = afwTable.SchemaMapper(peakMinimalSchema, schema)
        else:
            self.peakSchemaMapper = afwTable.SchemaMapper(peakSchema, schema)
            for item in peakSchema:
                if item.key not in peakMinimalSchema:
                    self.peakSchemaMapper.addMapping(item.key, item.field)
                    # Because SchemaMapper makes a copy of the output schema you give its ctor, it isn't
                    # updating this Schema in place.  That's probably a design flaw, but in the meantime,
                    # we'll keep that schema in sync with the peakSchemaMapper.getOutputSchema() manually,
                    # by adding the same fields to both.
                    schema.addField(item.field)
            assert schema == self.peakSchemaMapper.getOutputSchema(), "Logic bug mapping schemas"
        self.addSchemaKeys(schema)

    def addSchemaKeys(self, schema):
        self.nChildKey = schema.addField('deblend_nChild', type=np.int32,
                                         doc='Number of children this object has (defaults to 0)')
        self.psfKey = schema.addField('deblend_deblendedAsPsf', type='Flag',
                                      doc='Deblender thought this source looked like a PSF')
        self.psfCenterKey = afwTable.Point2DKey.addFields(schema, 'deblend_psfCenter',
                                                          'If deblended-as-psf, the PSF centroid', "pixel")
        self.psfFluxKey = schema.addField('deblend_psfFlux', type='D',
                                          doc='If deblended-as-psf, the PSF flux')
        self.tooManyPeaksKey = schema.addField('deblend_tooManyPeaks', type='Flag',
                                               doc='Source had too many peaks; '
                                               'only the brightest were included')
        self.tooBigKey = schema.addField('deblend_parentTooBig', type='Flag',
                                         doc='Parent footprint covered too many pixels')
        self.maskedKey = schema.addField('deblend_masked', type='Flag',
                                         doc='Parent footprint was predominantly masked')

        if self.config.catchFailures:
            self.deblendFailedKey = schema.addField('deblend_failed', type='Flag',
                                                    doc="Deblending failed on source")

        self.deblendSkippedKey = schema.addField('deblend_skipped', type='Flag',
                                                 doc="Deblender skipped this source")

        self.deblendRampedTemplateKey = schema.addField(
            'deblend_rampedTemplate', type='Flag',
            doc=('This source was near an image edge and the deblender used '
                 '"ramp" edge-handling.'))

        self.deblendPatchedTemplateKey = schema.addField(
            'deblend_patchedTemplate', type='Flag',
            doc=('This source was near an image edge and the deblender used '
                 '"patched" edge-handling.'))

        self.hasStrayFluxKey = schema.addField(
            'deblend_hasStrayFlux', type='Flag',
            doc=('This source was assigned some stray flux'))

        self.log.trace('Added keys to schema: %s', ", ".join(str(x) for x in (
                    self.nChildKey, self.psfKey, self.psfCenterKey, self.psfFluxKey,
                    self.tooManyPeaksKey, self.tooBigKey)))

    @pipeBase.timeMethod
    def run(self, exposure, sources):
        """!
        Get the psf from the provided exposure and then run deblend().

        @param[in]     exposure Exposure to process
        @param[in,out] sources  SourceCatalog containing sources detected on this exposure.

        @return None
        """
        psf = exposure.getPsf()
        assert sources.getSchema() == self.schema
        self.deblend(exposure, sources, psf)

    def _getPsfFwhm(self, psf, bbox):
        # It should be easier to get a PSF's fwhm;
        # https://dev.lsstcorp.org/trac/ticket/3030
        return psf.computeShape().getDeterminantRadius() * 2.35

    @pipeBase.timeMethod
    def deblend(self, exposure, srcs, psf):
        """!
        Deblend.

        @param[in]     exposure Exposure to process
        @param[in,out] srcs     SourceCatalog containing sources detected on this exposure.
        @param[in]     psf      PSF

        @return None
        """
        self.log.info("Deblending %d sources" % len(srcs))

        from lsst.meas.deblender.baseline import deblend

        # find the median stdev in the image...
        mi = exposure.getMaskedImage()
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setAndMask(mi.getMask().getPlaneBitMask(self.config.maskPlanes))
        stats = afwMath.makeStatistics(mi.getVariance(), mi.getMask(), afwMath.MEDIAN, statsCtrl)
        sigma1 = math.sqrt(stats.getValue(afwMath.MEDIAN))
        self.log.trace('sigma1: %g', sigma1)

        n0 = len(srcs)
        nparents = 0
        for i, src in enumerate(srcs):
            #t0 = time.clock()

            fp = src.getFootprint()
            pks = fp.getPeaks()

            # Since we use the first peak for the parent object, we should propagate its flags
            # to the parent source.
            src.assign(pks[0], self.peakSchemaMapper)

            if len(pks) < 2:
                continue

            if self.isLargeFootprint(fp):
                src.set(self.tooBigKey, True)
                self.skipParent(src, mi.getMask())
                self.log.trace('Parent %i: skipping large footprint', int(src.getId()))
                continue
            if self.isMasked(fp, exposure.getMaskedImage().getMask()):
                src.set(self.maskedKey, True)
                self.skipParent(src, mi.getMask())
                self.log.trace('Parent %i: skipping masked footprint', int(src.getId()))
                continue

            nparents += 1
            bb = fp.getBBox()
            psf_fwhm = self._getPsfFwhm(psf, bb)

            self.log.trace('Parent %i: deblending %i peaks', int(src.getId()), len(pks))

            self.preSingleDeblendHook(exposure, srcs, i, fp, psf, psf_fwhm, sigma1)
            npre = len(srcs)

            # This should really be set in deblend, but deblend doesn't have access to the src
            src.set(self.tooManyPeaksKey, len(fp.getPeaks()) > self.config.maxNumberOfPeaks)

            try:
                res = deblend(
                    fp, mi, psf, psf_fwhm, sigma1=sigma1,
                    psfChisqCut1=self.config.psfChisq1,
                    psfChisqCut2=self.config.psfChisq2,
                    psfChisqCut2b=self.config.psfChisq2b,
                    maxNumberOfPeaks=self.config.maxNumberOfPeaks,
                    strayFluxToPointSources=self.config.strayFluxToPointSources,
                    assignStrayFlux=self.config.assignStrayFlux,
                    strayFluxAssignment=self.config.strayFluxRule,
                    rampFluxAtEdge=(self.config.edgeHandling == 'ramp'),
                    patchEdges=(self.config.edgeHandling == 'noclip'),
                    tinyFootprintSize=self.config.tinyFootprintSize,
                    clipStrayFluxFraction=self.config.clipStrayFluxFraction,
                    weightTemplates=self.config.weightTemplates,
                    removeDegenerateTemplates=self.config.removeDegenerateTemplates,
                    maxTempDotProd=self.config.maxTempDotProd,
                    medianSmoothTemplate=self.config.medianSmoothTemplate
                )
                if self.config.catchFailures:
                    src.set(self.deblendFailedKey, False)
            except Exception as e:
                if self.config.catchFailures:
                    self.log.warn("Unable to deblend source %d: %s" % (src.getId(), e))
                    src.set(self.deblendFailedKey, True)
                    import traceback
                    traceback.print_exc()
                    continue
                else:
                    raise

            kids = []
            nchild = 0
            for j, peak in enumerate(res.deblendedParents[0].peaks):
                heavy = peak.getFluxPortion()
                if heavy is None or peak.skip:
                    src.set(self.deblendSkippedKey, True)
                    if not self.config.propagateAllPeaks:
                        # Don't care
                        continue
                    # We need to preserve the peak: make sure we have enough info to create a minimal
                    # child src
                    self.log.trace("Peak at (%i,%i) failed.  Using minimal default info for child.",
                                      pks[j].getIx(), pks[j].getIy())
                    if heavy is None:
                        # copy the full footprint and strip out extra peaks
                        foot = afwDet.Footprint(src.getFootprint())
                        peakList = foot.getPeaks()
                        peakList.clear()
                        peakList.append(peak.peak)
                        zeroMimg = afwImage.MaskedImageF(foot.getBBox())
                        heavy = afwDet.makeHeavyFootprint(foot, zeroMimg)
                    if peak.deblendedAsPsf:
                        if peak.psfFitFlux is None:
                            peak.psfFitFlux = 0.0
                        if peak.psfFitCenter is None:
                            peak.psfFitCenter = (peak.peak.getIx(), peak.peak.getIy())

                assert(len(heavy.getPeaks()) == 1)

                src.set(self.deblendSkippedKey, False)
                child = srcs.addNew()
                nchild += 1
                child.assign(heavy.getPeaks()[0], self.peakSchemaMapper)
                child.setParent(src.getId())
                child.setFootprint(heavy)
                child.set(self.psfKey, peak.deblendedAsPsf)
                child.set(self.hasStrayFluxKey, peak.strayFlux is not None)
                if peak.deblendedAsPsf:
                    (cx, cy) = peak.psfFitCenter
                    child.set(self.psfCenterKey, afwGeom.Point2D(cx, cy))
                    child.set(self.psfFluxKey, peak.psfFitFlux)
                child.set(self.deblendRampedTemplateKey, peak.hasRampedTemplate)
                child.set(self.deblendPatchedTemplateKey, peak.patched)
                kids.append(child)

            # Child footprints may extend beyond the full extent of their parent's which
            # results in a failure of the replace-by-noise code to reinstate these pixels
            # to their original values.  The following updates the parent footprint
            # in-place to ensure it contains the full union of itself and all of its
            # children's footprints.
            spans = src.getFootprint().spans
            for child in kids:
                spans = spans.union(child.getFootprint().spans)
            src.getFootprint().setSpans(spans)

            src.set(self.nChildKey, nchild)

            self.postSingleDeblendHook(exposure, srcs, i, npre, kids, fp, psf, psf_fwhm, sigma1, res)
            #print 'Deblending parent id', src.getId(), 'took', time.clock() - t0

        n1 = len(srcs)
        self.log.info('Deblended: of %i sources, %i were deblended, creating %i children, total %i sources'
                      % (n0, nparents, n1-n0, n1))

    def preSingleDeblendHook(self, exposure, srcs, i, fp, psf, psf_fwhm, sigma1):
        pass

    def postSingleDeblendHook(self, exposure, srcs, i, npre, kids, fp, psf, psf_fwhm, sigma1, res):
        pass

    def isLargeFootprint(self, footprint):
        """Returns whether a Footprint is large

        'Large' is defined by thresholds on the area, size and axis ratio.
        These may be disabled independently by configuring them to be non-positive.

        This is principally intended to get rid of satellite streaks, which the
        deblender or other downstream processing can have trouble dealing with
        (e.g., multiple large HeavyFootprints can chew up memory).
        """
        if self.config.maxFootprintArea > 0 and footprint.getArea() > self.config.maxFootprintArea:
            return True
        if self.config.maxFootprintSize > 0:
            bbox = footprint.getBBox()
            if max(bbox.getWidth(), bbox.getHeight()) > self.config.maxFootprintSize:
                return True
        if self.config.minFootprintAxisRatio > 0:
            axes = afwEll.Axes(footprint.getShape())
            if axes.getB() < self.config.minFootprintAxisRatio*axes.getA():
                return True
        return False

    def isMasked(self, footprint, mask):
        """Returns whether the footprint violates the mask limits"""
        size = float(footprint.getArea())
        for maskName, limit in self.config.maskLimits.items():
            maskVal = mask.getPlaneBitMask(maskName)
            unmaskedSpan = footprint.spans.intersectNot(mask, maskVal) # spanset of unmasked pixels
            if (size - unmaskedSpan.getArea())/size > limit:
                return True
        return False

    def skipParent(self, source, mask):
        """Indicate that the parent source is not being deblended

        We set the appropriate flags and mask.

        @param source  The source to flag as skipped
        @param mask  The mask to update
        """
        fp = source.getFootprint()
        source.set(self.deblendSkippedKey, True)
        source.set(self.nChildKey, len(fp.getPeaks())) # It would have this many if we deblended them all
        if self.config.notDeblendedMask:
            mask.addMaskPlane(self.config.notDeblendedMask)
            fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))

class MultibandDeblendConfig(pexConfig.Config):
    """MultibandDeblendConfig

    Configuration for the multiband deblender.
    The parameters are organized by the parameter types, which are
    - Stopping Criteria: Used to determine if the fit has converged
    - Position Fitting Criteria: Used to fit the positions of the peaks
    - Constraints: Used to apply constraints to the peaks and their components
    - Other: Parameters that don't fit into the above categories
    """
    # Stopping Criteria
    maxIter = pexConfig.Field(dtype=int, default=200,
                              doc=("Maximum number of iterations to deblend a single parent"))
    relativeError = pexConfig.Field(dtype=float, default=1e-3,
                                    doc=("Relative error to use when determining stopping criteria"))

    # Blend Configuration options
    minTranslation = pexConfig.Field(dtype=float, default=1e-3,
                                     doc=("A peak must be updated by at least 'minTranslation' (pixels)"
                                          "or no update is performed."
                                          "This field is ignored if fitPositions is False."))
    refinementSkip = pexConfig.Field(dtype=int, default=10,
                                      doc=("If fitPositions is True, the positions and box sizes are"
                                           "updated on every 'refinementSkip' iterations."))
    translationMethod = pexConfig.Field(dtype=str, default="default",
                                        doc=("Method to use for fitting translations."
                                             "Currently 'default' is the only available option,"
                                             "which performs a linear fit, but it is possible that we"
                                             "will use galsim or some other method as a future option"))
    edgeFluxThresh = pexConfig.Field(dtype=float, default=1.0,
                                     doc=("Boxes are resized when the flux at an edge is "
                                          "> edgeFluxThresh * background RMS"))
    exactLipschitz = pexConfig.Field(dtype=bool, default=False,
                                     doc=("Calculate exact Lipschitz constant in every step"
                                          "(True) or only calculate the approximate"
                                          "Lipschitz constant with significant changes in A,S"
                                          "(False)"))
    stepSlack = pexConfig.Field(dtype=float, default=0.2,
                                doc=("A fractional measure of how much a value (like the exactLipschitz)"
                                     "can change before it needs to be recalculated."
                                     "This must be between 0 and 1."))

    # Constraints
    constraints = pexConfig.Field(dtype=str, default="1,+,S,M",
                                  doc=("List of constraints to use for each object"
                                       "(order does not matter)"
                                       "Current options are all used by default:\n"
                                       "S: symmetry\n"
                                       "M: monotonicity\n"
                                       "1: normalized SED to unity"
                                       "+: non-negative morphology"))
    symmetryThresh = pexConfig.Field(dtype=float, default=1.0,
                                     doc=("Strictness of symmetry, from"
                                          "0 (no symmetry enforced) to"
                                          "1 (perfect symmetry required)."
                                          "If 'S' is not in `constraints`, this argument is ignored"))
    l0Thresh = pexConfig.Field(dtype=float, default=np.nan,
                               doc=("L0 threshold. NaN results in no L0 penalty."))
    l1Thresh = pexConfig.Field(dtype=float, default=np.nan,
                               doc=("L1 threshold. NaN results in no L1 penalty."))
    tvxThresh = pexConfig.Field(dtype=float, default=np.nan,
                                doc=("Threshold for TV (total variation) constraint in the x-direction."
                                     "NaN results in no TVx penalty."))
    tvyThresh = pexConfig.Field(dtype=float, default=np.nan,
                                doc=("Threshold for TV (total variation) constraint in the y-direction."
                                     "NaN results in no TVy penalty."))

    # Other scarlet paremeters
    useWeights = pexConfig.Field(dtype=bool, default=False, doc="Use inverse variance as deblender weights")
    bgScale = pexConfig.Field(dtype=float, default=0.5,
                              doc=("Fraction of background RMS level to use as a"
                                   "cutoff for defining the background of the image"
                                   "This is used to initialize the model for each source"
                                   "and to set the size of the bounding box for each source"
                                   "every `refinementSkip` iteration."))
    usePsfConvolution = pexConfig.Field(dtype=bool, default=True,
                                        doc=("Whether or not to convolve the morphology with the"
                                             "PSF in each band or use the same morphology"
                                             "in all bands"))
    saveTemplates = pexConfig.Field(dtype=bool, default=True,
                                    doc="Whether or not to save the SEDs and templates")
    processSingles = pexConfig.Field(dtype=bool, default=False,
                                     doc="Whether or not to process isolated sources in the deblender")
    badMask = pexConfig.Field(dtype=str, default="BAD,CR,NO_DATA,SAT,SUSPECT",
                              doc="Whether or not to process isolated sources in the deblender")
    # Old deblender parameters used in this implementation (some of which might be removed later)

    maxNumberOfPeaks = pexConfig.Field(dtype=int, default=0,
                                     doc=("Only deblend the brightest maxNumberOfPeaks peaks in the parent"
                                          " (<= 0: unlimited)"))
    maxFootprintArea = pexConfig.Field(dtype=int, default=1000000,
                                     doc=("Maximum area for footprints before they are ignored as large; "
                                          "non-positive means no threshold applied"))
    maxFootprintSize = pexConfig.Field(dtype=int, default=0,
                                     doc=("Maximum linear dimension for footprints before they are ignored "
                                          "as large; non-positive means no threshold applied"))
    minFootprintAxisRatio = pexConfig.Field(dtype=float, default=0.0,
                                          doc=("Minimum axis ratio for footprints before they are ignored "
                                               "as large; non-positive means no threshold applied"))
    notDeblendedMask = pexConfig.Field(dtype=str, default="NOT_DEBLENDED", optional=True,
                                     doc="Mask name for footprints not deblended, or None")

    tinyFootprintSize = pexConfig.RangeField(dtype=int, default=2, min=2, inclusiveMin=True,
                                           doc=('Footprints smaller in width or height than this value will '
                                                'be ignored; minimum of 2 due to PSF gradient calculation.'))
    catchFailures = pexConfig.Field(dtype=bool, default=False,
                                  doc=("If True, catch exceptions thrown by the deblender, log them, "
                                       "and set a flag on the parent, instead of letting them propagate up"))
    propagateAllPeaks = pexConfig.Field(dtype=bool, default=False,
                                      doc=('Guarantee that all peaks produce a child source.'))
    maskPlanes = pexConfig.ListField(dtype=str, default=["SAT", "INTRP", "NO_DATA"],
                                   doc="Mask planes to ignore when performing statistics")
    maskLimits = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={},
        doc=("Mask planes with the corresponding limit on the fraction of masked pixels. "
             "Sources violating this limit will not be deblended."),
    )

    edgeHandling = pexConfig.ChoiceField(
        doc='What to do when a peak to be deblended is close to the edge of the image',
        dtype=str, default='ramp',
        allowed={
            'clip': 'Clip the template at the edge AND the mirror of the edge.',
            'ramp': 'Ramp down flux at the image edge by the PSF',
            'noclip': 'Ignore the edge when building the symmetric template.',
        }
    )

    medianSmoothTemplate = pexConfig.Field(dtype=bool, default=False,
                                         doc="Apply a smoothing filter to all of the template images")
    medianFilterHalfsize = pexConfig.Field(dtype=float, default=2,
                                         doc=('Half size of the median smoothing filter'))
    clipFootprintToNonzero = pexConfig.Field(dtype=bool, default=True,
                                             doc=("Clip non-zero spans in the footprints"))

    conserveFlux = pexConfig.Field(dtype=bool, default=False,
                                   doc=("Reapportion flux to the footprints so that flux is conserved"))
    weightTemplates = pexConfig.Field(dtype=bool, default=False,
                                    doc=("If true, a least-squares fit of the templates will be done to the "
                                         "full image. The templates will be re-weighted based on this fit."))
    strayFluxToPointSources = pexConfig.ChoiceField(
        doc='When the deblender should attribute stray flux to point sources',
        dtype=str, default='necessary',
        allowed={
            'necessary': 'When there is not an extended object in the footprint',
            'always': 'Always',
            'never': ('Never; stray flux will not be attributed to any deblended child '
                      'if the deblender thinks all peaks look like point sources'),
        }
    )

    assignStrayFlux = pexConfig.Field(dtype=bool, default=True,
                                    doc='Assign stray flux (not claimed by any child in the deblender) '
                                        'to deblend children.')

    strayFluxRule = pexConfig.ChoiceField(
        doc='How to split flux among peaks',
        dtype=str, default='trim',
        allowed={
            'r-to-peak': '~ 1/(1+R^2) to the peak',
            'r-to-footprint': ('~ 1/(1+R^2) to the closest pixel in the footprint.  '
                               'CAUTION: this can be computationally expensive on large footprints!'),
            'nearest-footprint': ('Assign 100% to the nearest footprint (using L-1 norm aka '
                                  'Manhattan distance)'),
            'trim': ('Shrink the parent footprint to pixels that are not assigned to children')
        }
    )

    clipStrayFluxFraction = pexConfig.Field(dtype=float, default=0.001,
                                          doc=('When splitting stray flux, clip fractions below '
                                               'this value to zero.'))
    getTemplateSum = pexConfig.Field(dtype=bool, default=False,
                                     doc=("As part of the flux calculation, the sum of the templates is"
                                          "calculated. If 'getTemplateSum==True' then the sum of the"
                                          "templates is stored in the result (a 'PerFootprint')."))

class MultibandDeblendTask(pipeBase.Task):
    """MultibandDeblendTask

    Split blended sources into individual sources.

    This task has no return value; it only modifies the SourceCatalog in-place.
    """
    ConfigClass = MultibandDeblendConfig
    _DefaultName = "multibandDeblend"

    def __init__(self, schema, peakSchema=None, **kwargs):
        """Create the task, adding necessary fields to the given schema.

        Parameters
        ----------
        schema: `lsst.afw.table.schema.schema.Schema`
            Schema object for measurement fields; will be modified in-place.
        peakSchema: `lsst.afw.table.schema.schema.Schema`
            Schema of Footprint Peaks that will be passed to the deblender.
            Any fields beyond the PeakTable minimal schema will be transferred
            to the main source Schema.  If None, no fields will be transferred
            from the Peaks.
        bands: list of str
            Names of the filters used for the eposures. This is needed to store the SED as a field
        **kwargs
            Passed to Task.__init__.
        """
        from lsst.meas.deblender import plugins
        import scarlet

        pipeBase.Task.__init__(self, **kwargs)
        if not self.config.conserveFlux and not self.config.saveTemplates:
            raise ValueError("Either `conserveFlux` or `saveTemplates` must be True")

        peakMinimalSchema = afwDet.PeakTable.makeMinimalSchema()
        if peakSchema is None:
            # In this case, the peakSchemaMapper will transfer nothing, but we'll still have one
            # to simplify downstream code
            self.peakSchemaMapper = afwTable.SchemaMapper(peakMinimalSchema, schema)
        else:
            self.peakSchemaMapper = afwTable.SchemaMapper(peakSchema, schema)
            for item in peakSchema:
                if item.key not in peakMinimalSchema:
                    self.peakSchemaMapper.addMapping(item.key, item.field)
                    # Because SchemaMapper makes a copy of the output schema you give its ctor, it isn't
                    # updating this Schema in place.  That's probably a design flaw, but in the meantime,
                    # we'll keep that schema in sync with the peakSchemaMapper.getOutputSchema() manually,
                    # by adding the same fields to both.
                    schema.addField(item.field)
            assert schema == self.peakSchemaMapper.getOutputSchema(), "Logic bug mapping schemas"
        self._addSchemaKeys(schema)

        # Create the plugins for multiband deblending using the Config options

        # Basic deblender configuration
        config = scarlet.config.Config(
            center_min_dist=self.config.minTranslation,
            edge_flux_thresh=self.config.edgeFluxThresh,
            exact_lipschitz=self.config.exactLipschitz,
            refine_skip=self.config.refinementSkip,
            slack=self.config.stepSlack,
        )
        if self.config.translationMethod != "default":
            err = "Currently the only supported translationMethod is 'default', you entered '{0}'"
            raise NotImplementedError(err.format(self.config.translationMethod))

        # If the default constraints are not used, set the constraints for
        # all of the sources
        constraints = None
        _constraints = self.config.constraints.split(",")
        if (sorted(_constraints) != ['+', '1', 'M', 'S']
            or ~np.isnan(self.config.l0Thresh)
            or ~np.isnan(self.config.l1Thresh)
        ):
            constraintDict = {
                "+": scarlet.constraints.PositivityConstraint,
                "1": scarlet.constraints.SimpleConstraint,
                "M": scarlet.constraints.DirectMonotonicityConstraint(use_nearest=False),
                "S": scarlet.constraints.DirectSymmetryConstraint(sigma=self.config.symmetryThresh)
            }
            for c in _constraints:
                if constraints is None:
                    constraints = constraintDict[c]
                else:
                    constraints = constraints & constraintDict[c]
            if constraints is None:
                constraints = scarlet.constraints.MinimalConstraint()
            if ~np.isnan(self.config.l0Thresh):
                constraints = constraints & scarlet.constraints.L0Constraint(self.config.l0Thresh)
            if ~np.isnan(self.config.l1Thresh):
                constraints = constraints & scarlet.constraints.L1Constraint(self.config.l1Thresh)
            if ~np.isnan(self.config.tvxThresh):
                constraints = constraints & scarlet.constraints.TVxConstraint(self.config.tvxThresh)
            if ~np.isnan(self.config.tvyThresh):
                constraints = constraints & scarlet.constraints.TVyConstraint(self.config.tvyThresh)

        multiband_plugin = plugins.DeblenderPlugin(
            plugins.buildMultibandTemplates,
            useWeights=self.config.useWeights,
            usePsf=self.config.usePsfConvolution,
            constraints=constraints,
            config=config,
            maxIter=self.config.maxIter,
            bgScale=self.config.bgScale,
            relativeError=self.config.relativeError,
            badMask=self.config.badMask.split(","),
        )
        self.plugins = [multiband_plugin]

        # Plugins from the old deblender for post-template processing
        # (see lsst.meas_deblender.baseline.deblend)
        patchEdges = self.config.edgeHandling == 'noclip'
        if self.config.edgeHandling == 'ramp':
            self.plugins.append(plugins.DeblenderPlugin(plugins.rampFluxAtEdge, patchEdges=False))
        if self.config.medianSmoothTemplate:
            self.plugins.append(plugins.DeblenderPlugin(plugins.medianSmoothTemplates,
                                                      medianFilterHalfsize=self.config.medianFilterHalfsize))
        if self.config.clipFootprintToNonzero:
            self.plugins.append(plugins.DeblenderPlugin(plugins.clipFootprintsToNonzero))
        if self.config.conserveFlux:
            if self.config.weightTemplates:
                self.plugins.append(plugins.DeblenderPlugin(plugins.weightTemplates))
            self.plugins.append(plugins.DeblenderPlugin(plugins.apportionFlux,
                                                clipStrayFluxFraction=self.config.clipStrayFluxFraction,
                                                assignStrayFlux=self.config.assignStrayFlux,
                                                strayFluxAssignment=self.config.strayFluxRule,
                                                strayFluxToPointSources=self.config.strayFluxToPointSources,
                                                getTemplateSum=self.config.getTemplateSum))


    def _addSchemaKeys(self, schema):
        """Add deblender specific keys to the schema
        """
        self.runtimeKey = schema.addField('runtime', type=np.float32, doc='runtime in ms')
        # Keys from old Deblender that might be kept in the new deblender
        self.nChildKey = schema.addField('deblend_nChild', type=np.int32,
                                         doc='Number of children this object has (defaults to 0)')
        self.psfKey = schema.addField('deblend_deblendedAsPsf', type='Flag',
                                      doc='Deblender thought this source looked like a PSF')
        self.tooManyPeaksKey = schema.addField('deblend_tooManyPeaks', type='Flag',
                                               doc='Source had too many peaks; '
                                               'only the brightest were included')
        self.tooBigKey = schema.addField('deblend_parentTooBig', type='Flag',
                                         doc='Parent footprint covered too many pixels')
        self.maskedKey = schema.addField('deblend_masked', type='Flag',
                                         doc='Parent footprint was predominantly masked')
        self.deblendFailedKey = schema.addField('deblend_failed', type='Flag',
                                                 doc="Deblending failed on source")

        self.deblendSkippedKey = schema.addField('deblend_skipped', type='Flag',
                                                 doc="Deblender skipped this source")

        # Keys from the old Deblender that are likely to be removed for the new deblender
        # TODO: Remove these if they remain unused
        self.psfCenterKey = afwTable.Point2DKey.addFields(schema, 'deblend_psfCenter',
                                                          'If deblended-as-psf, the PSF centroid', "pixel")
        self.psfFluxKey = schema.addField('deblend_psfFlux', type='D',
                                          doc='If deblended-as-psf, the PSF flux')
        self.deblendRampedTemplateKey = schema.addField(
            'deblend_rampedTemplate', type='Flag',
            doc=('This source was near an image edge and the deblender used '
                 '"ramp" edge-handling.'))

        self.deblendPatchedTemplateKey = schema.addField(
            'deblend_patchedTemplate', type='Flag',
            doc=('This source was near an image edge and the deblender used '
                 '"patched" edge-handling.'))

        self.hasStrayFluxKey = schema.addField(
            'deblend_hasStrayFlux', type='Flag',
            doc=('This source was assigned some stray flux'))

        self.log.trace('Added keys to schema: %s', ", ".join(str(x) for x in (
                    self.nChildKey, self.psfKey, self.psfCenterKey, self.psfFluxKey,
                    self.tooManyPeaksKey, self.tooBigKey)))

    @pipeBase.timeMethod
    def run(self, exposures, sources):
        """Get the psf from each exposure and then run deblend().

        Parameters
        ----------
        exposures: dict
            Keys of the dict are the names of the filters and values are
            `lsst.afw.image.exposure.exposure.ExposureF`'s.
            The exposures should be co-added images of the same
            shape and region of the sky.
        sources: dict
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s, which
            should be a merged catalog of the sources in each band.

        Returns
        -------
        flux_catalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are the flux-conserved catalogs with heavy footprints with
            the image data weighted by the multiband templates.
            If `self.config.conserveFlux` is `False`, then this item will be None
        template_catalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
            If `self.config.saveTemplates` is `False`, then this item will be None
        """
        psfs = {B:exp.getPsf() for B, exp in exposures.items()}
        return self.deblend(exposures, sources, psfs)

    def _getPsfFwhm(self, psf, bbox):
        return psf.computeShape().getDeterminantRadius() * 2.35

    def _addChild(self, parentId, peak, sources, heavy):
        """Add a child to a catalog

        This creates a new child in the source catalog,
        assigning it a parent id, adding a footprint,
        and setting all appropriate flags based on the
        deblender result.
        """
        assert len(heavy.getPeaks())==1
        src = sources.addNew()
        src.assign(heavy.getPeaks()[0], self.peakSchemaMapper)
        src.setParent(parentId)
        src.setFootprint(heavy)
        src.set(self.psfKey, peak.deblendedAsPsf)
        src.set(self.hasStrayFluxKey, peak.strayFlux is not None)
        src.set(self.deblendRampedTemplateKey, peak.hasRampedTemplate)
        src.set(self.deblendPatchedTemplateKey, peak.patched)
        src.set(self.runtimeKey, 0)
        return src

    @pipeBase.timeMethod
    def deblend(self, exposures, sources, psfs, bands=None):
        """Deblend a data cube of multiband images

        Parameters
        ----------
        exposures: dict
            Keys of the dict are the names of the filters and values are
            `lsst.afw.image.exposure.exposure.ExposureF`'s.
            The exposures should be co-added images of the same
            shape and region of the sky.
        sources: dict
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s, which
            should be a merged catalog of the sources in each band ('deepCoadd_mergeDet').
        psfs: dict
        bands: list, default=None
            Names of the bands in the deblender.
            If `bands` is `None`, the keys of `exposures` are used.
            Either `bands` should be specified or `exposures` should be an
            `OrderedDict` to set the preferential order of the filters.

        Returns
        -------
        flux_catalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are the flux-conserved catalogs with heavy footprints with
            the image data weighted by the multiband templates.
            If `self.config.conserveFlux` is `False`, then this item will be None
        template_catalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
            If `self.config.saveTemplates` is `False`, then this item will be None
        """
        from lsst.meas.deblender.baseline import newDeblend
        import deblender

        if bands is None:
            bands = list(exposures.keys())
        maskedImages = {band:exp.getMaskedImage() for band, exp in exposures.items()}
        self.log.info("Deblending {0} sources in {1} exposures".format(len(sources), len(bands)))

        # find the median stdev in each image
        sigmas = {}
        for f, exposure in exposures.items():
            mi = exposure.getMaskedImage()
            statsCtrl = afwMath.StatisticsControl()
            statsCtrl.setAndMask(mi.getMask().getPlaneBitMask(self.config.maskPlanes))
            stats = afwMath.makeStatistics(mi.getVariance(), mi.getMask(), afwMath.MEDIAN, statsCtrl)
            sigma1 = math.sqrt(stats.getValue(afwMath.MEDIAN))
            self.log.trace('Exposure {0}, sigma1: {1}'.format(f, sigma1))
            sigmas[f] = sigma1

        # Create the output catalogs
        if self.config.conserveFlux:
            flux_catalogs = {band:afwTable.SourceCatalog(sources.clone()) for band in bands}
        else:
            flux_catalogs = None
        if self.config.saveTemplates:
            template_catalogs = {band:afwTable.SourceCatalog(sources.clone()) for band in bands}
        else:
            template_catalogs = None

        n0 = len(sources)
        nparents = 0
        maskedImages = {band: exp.getMaskedImage() for band, exp in exposures.items()}
        for pk, src in enumerate(sources):
            foot = src.getFootprint()
            logger.info("id: {0}".format(src["id"]))
            peaks = foot.getPeaks()

            # Since we use the first peak for the parent object, we should propagate its flags
            # to the parent source.
            src.assign(peaks[0], self.peakSchemaMapper)

            # Block of Skipping conditions
            if len(peaks) < 2 and not self.config.processSingles:
                for band in bands:
                    if self.config.saveTemplates:
                        tsrc = template_catalogs[band].addNew()
                        tsrc.assign(src)
                        tsrc.set(self.runtimeKey, 0)
                        templateParents[band] = tsrc
                    if self.config.conserveFlux:
                        tsrc = flux_catalogs[band].addNew()
                        tsrc.assign(src)
                        tsrc.set(self.runtimeKey, 0)
                        fluxParents[band] = tsrc
                continue
            if self.isLargeFootprint(foot):
                src.set(self.tooBigKey, True)
                self.skipParent(src, [mi.getMask() for mi in maskedImages])
                self.log.trace('Parent %i: skipping large footprint', int(src.getId()))
                continue
            if self.isMasked(foot, exposure.getMaskedImage().getMask()):
                src.set(self.maskedKey, True)
                self.skipParent(src, mi.getMask())
                self.log.trace('Parent %i: skipping masked footprint', int(src.getId()))
                continue
            if len(peaks) > self.config.maxNumberOfPeaks:
                src.set(self.tooManyPeaksKey, True)
                msg = 'Parent {0}: Too many peaks, using the first {1} peaks'
                self.log.trace(msg.format(int(src.getId()), self.config.maxNumberOfPeaks))

            nparents += 1
            bbox = foot.getBBox()
            psf_fwhms = {band:self._getPsfFwhm(psf, bbox) for band, psf in psfs.items()}
            self.log.trace('Parent %i: deblending %i peaks', int(src.getId()), len(peaks))
            self.preSingleDeblendHook(exposures, sources, pk, foot, psfs, psf_fwhms, sigmas)
            npre = len(sources)
            # Run the deblender
            try:
                t0=time.time()
                PARENT = afwImage.PARENT
                # Build the parameter lists with the same ordering
                images = [maskedImages[band].Factory(maskedImages[band], bbox, PARENT)
                              for band in bands]
                psf_list = [psfs[band] for band in bands]
                fwhm_list = [psf_fwhms[band] for band in bands]
                avgNoise = [sigmas[band] for band in bands]

                result = newDeblend(debPlugins=self.plugins,
                                    footprint=foot,
                                    maskedImages=images,
                                    psfs=psf_list,
                                    psfFwhms=fwhm_list,
                                    filters=bands,
                                    avgNoise=avgNoise,
                                    maxNumberOfPeaks=self.config.maxNumberOfPeaks
                )
                tf=time.time()
                runtime = (tf-t0)*1000
                if result.failed:
                    src.set(self.deblendFailedKey, False)
                    src.set(self.runtimeKey, 0)
                    continue
            except Exception as e:
                if self.config.catchFailures:
                    self.log.warn("Unable to deblend source %d: %s" % (src.getId(), e))
                    src.set(self.deblendFailedKey, True)
                    src.set(self.runtimeKey, 0)
                    import traceback
                    traceback.print_exc()
                    continue
                else:
                    raise

            # Add the merged source as a parent in the catalog for each band
            templateParents = {}
            fluxParents = {}
            parentId = src.getId()
            for band in bands:
                if self.config.saveTemplates:
                    tsrc = template_catalogs[band].addNew()
                    tsrc.assign(src)
                    tsrc.set("id", parentId)
                    tsrc.set(self.runtimeKey, runtime)
                    _fp = afwDet.Footprint()
                    _fp.setPeakSchema(src.getFootprint().getPeaks().getSchema())
                    tsrc.setFootprint(_fp)
                    templateParents[band] = tsrc
                if self.config.conserveFlux:
                    tsrc = flux_catalogs[band].addNew()
                    tsrc.assign(src)
                    tsrc.set(self.runtimeKey, runtime)
                    tsrc.set("id", parentId)
                    _fp = afwDet.Footprint()
                    _fp.setPeakSchema(src.getFootprint().getPeaks().getSchema())
                    tsrc.setFootprint(_fp)
                    fluxParents[band] = tsrc

            # Add each source to the catalogs in each band
            templateSpans = {band:afwGeom.SpanSet() for band in bands}
            fluxSpans = {band:afwGeom.SpanSet() for band in bands}
            nchild = 0
            for j, multiPeak in enumerate(result.peaks):
                heavy = {band:peak.getFluxPortion() for band, peak in multiPeak.deblendedPeaks.items()}
                no_flux = all([v is None for v in heavy.values()])
                skip_peak = all([peak.skip for peak in multiPeak.deblendedPeaks.values()])
                if no_flux or skip_peak:
                    src.set(self.deblendSkippedKey, True)
                    if not self.config.propagateAllPeaks:
                        # We don't care
                        continue
                    # We need to preserve the peak: make sure we have enough info to create a minimal
                    # child src
                    msg = "Peak at {0} failed deblending.  Using minimal default info for child."
                    self.log.trace(msg.format(multiPeak.x, multiPeak.y))

                    # copy the full footprint and strip out extra peaks
                    pfoot = afwDet.Footprint(foot)
                    peakList = pfoot.getPeaks()
                    peakList.clear()
                    pfoot.addPeak(multiPeak.x, multiPeak.y, 0)
                    zeroMimg = afwImage.MaskedImageF(pfoot.getBBox())
                    for band in bands:
                        heavy[band] = afwDet.makeHeavyFootprint(pfoot, zeroMimg)
                else:
                    src.set(self.deblendSkippedKey, False)

                # Add the peak to the source catalog in each band
                for band in bands:
                    if len(heavy[band].getPeaks()) != 1:
                        raise ValueError("Heavy footprint has multiple peaks, expected 1")
                    peak = multiPeak.deblendedPeaks[band]
                    if self.config.saveTemplates:
                        cat = template_catalogs[band]
                        tfoot = peak.templateFootprint
                        timg  = afwImage.MaskedImageF(peak.templateImage)
                        tHeavy = afwDet.makeHeavyFootprint(tfoot, timg)
                        child = self._addChild(parentId, peak, cat, tHeavy)
                        if parentId==0:
                            child.setId(src.getId())
                            child.set(self.runtimeKey, runtime)
                        else:
                            _peak = tHeavy.getPeaks()[0]
                            templateParents[band].getFootprint().addPeak(_peak.getFx(), _peak.getFy(),
                                                                         _peak.getPeakValue())
                            templateSpans[band] = templateSpans[band].union(tHeavy.getSpans())
                    if self.config.conserveFlux:
                        cat = flux_catalogs[band]
                        child = self._addChild(parentId, peak, cat, heavy[band])
                        if parentId==0:
                            child.setId(src.getId())
                            child.set(self.runtimeKey, runtime)
                        else:
                            _peak = heavy[band].getPeaks()[0]
                            fluxParents[band].getFootprint().addPeak(_peak.getFx(), _peak.getFy(),
                                                                     _peak.getPeakValue())
                            fluxSpans[band] = fluxSpans[band].union(heavy[band].getSpans())
                    nchild += 1

            # Child footprints may extend beyond the full extent of their parent's which
            # results in a failure of the replace-by-noise code to reinstate these pixels
            # to their original values.  The following updates the parent footprint
            # in-place to ensure it contains the full union of itself and all of its
            # children's footprints.
            for band in bands:
                if self.config.saveTemplates:
                    templateParents[band].set(self.nChildKey, nchild)
                    templateParents[band].getFootprint().setSpans(templateSpans[band])
                if self.config.conserveFlux:
                    fluxParents[band].set(self.nChildKey, nchild)
                    fluxParents[band].getFootprint().setSpans(fluxSpans[band])

            self.postSingleDeblendHook(exposure, flux_catalogs, template_catalogs,
                                       pk, npre, foot, psfs, psf_fwhms, sigmas, result)

        if flux_catalogs is not None:
            n1 = len(list(flux_catalogs.values())[0])
        else:
            n1 = len(list(template_catalogs.values())[0])
        self.log.info('Deblended: of %i sources, %i were deblended, creating %i children, total %i sources'
                      % (n0, nparents, n1-n0, n1))
        return flux_catalogs, template_catalogs

    def preSingleDeblendHook(self, exposures, sources, pk, fp, psfs, psf_fwhms, sigmas):
        pass

    def postSingleDeblendHook(self, exposures, flux_catalogs, template_catalogs,
                              pk, npre, fp, psfs, psf_fwhms, sigmas, result):
        pass

    def isLargeFootprint(self, footprint):
        """Returns whether a Footprint is large

        'Large' is defined by thresholds on the area, size and axis ratio.
        These may be disabled independently by configuring them to be non-positive.

        This is principally intended to get rid of satellite streaks, which the
        deblender or other downstream processing can have trouble dealing with
        (e.g., multiple large HeavyFootprints can chew up memory).
        """
        if self.config.maxFootprintArea > 0 and footprint.getArea() > self.config.maxFootprintArea:
            return True
        if self.config.maxFootprintSize > 0:
            bbox = footprint.getBBox()
            if max(bbox.getWidth(), bbox.getHeight()) > self.config.maxFootprintSize:
                return True
        if self.config.minFootprintAxisRatio > 0:
            axes = afwEll.Axes(footprint.getShape())
            if axes.getB() < self.config.minFootprintAxisRatio*axes.getA():
                return True
        return False

    def isMasked(self, footprint, mask):
        """Returns whether the footprint violates the mask limits"""
        size = float(footprint.getArea())
        for maskName, limit in self.config.maskLimits.items():
            maskVal = mask.getPlaneBitMask(maskName)
            unmaskedSpan = footprint.spans.intersectNot(mask, maskVal) # spanset of unmasked pixels
            if (size - unmaskedSpan.getArea())/size > limit:
                return True
        return False

    def skipParent(self, source, masks):
        """Indicate that the parent source is not being deblended

        We set the appropriate flags and masks for each exposure.

        Parameters
        ----------
        source: `lsst.afw.table.source.source.SourceRecord`
            The source to flag as skipped
        masks: list of `lsst.afw.image.mask.mask.MaskX`
            The mask in each band to update with the non-detection
        """
        fp = source.getFootprint()
        source.set(self.deblendSkippedKey, True)
        source.set(self.nChildKey, len(fp.getPeaks())) # It would have this many if we deblended them all
        if self.config.notDeblendedMask:
            for mask in masks:
                mask.addMaskPlane(self.config.notDeblendedMask)
                fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))
