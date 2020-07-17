# This file is part of meas_deblender.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import numpy as np

import lsst.log
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable

logger = lsst.log.Log.getLogger("meas.deblender.deblend")

__all__ = 'SourceDeblendConfig', 'SourceDeblendTask'


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
                                             doc=('Footprints smaller in width or height than this value '
                                                  'will be ignored; minimum of 2 due to PSF gradient '
                                                  'calculation.'))

    propagateAllPeaks = pexConfig.Field(dtype=bool, default=False,
                                        doc=('Guarantee that all peaks produce a child source.'))
    catchFailures = pexConfig.Field(
        dtype=bool, default=False,
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
    weightTemplates = pexConfig.Field(
        dtype=bool, default=False,
        doc=("If true, a least-squares fit of the templates will be done to the "
             "full image. The templates will be re-weighted based on this fit."))
    removeDegenerateTemplates = pexConfig.Field(dtype=bool, default=False,
                                                doc=("Try to remove similar templates?"))
    maxTempDotProd = pexConfig.Field(
        dtype=float, default=0.5,
        doc=("If the dot product between two templates is larger than this value, we consider them to be "
             "describing the same object (i.e. they are degenerate).  If one of the objects has been "
             "labeled as a PSF it will be removed, otherwise the template with the lowest value will "
             "be removed."))
    medianSmoothTemplate = pexConfig.Field(dtype=bool, default=True,
                                           doc="Apply a smoothing filter to all of the template images")

    # Testing options
    # Some obs packages and ci packages run the full pipeline on a small
    # subset of data to test that the pipeline is functioning properly.
    # This is not meant as scientific validation, so it can be useful
    # to only run on a small subset of the data that is large enough to
    # test the desired pipeline features but not so long that the deblender
    # is the tall pole in terms of execution times.
    useCiLimits = pexConfig.Field(
        dtype=bool, default=False,
        doc="Limit the number of sources deblended for CI to prevent long build times")
    ciDeblendChildRange = pexConfig.ListField(
        dtype=int, default=[2, 10],
        doc="Only deblend parent Footprints with a number of peaks in the (inclusive) range indicated."
            "If `useCiLimits==False` then this parameter is ignored.")
    ciNumParentsToDeblend = pexConfig.Field(
        dtype=int, default=10,
        doc="Only use the first `ciNumParentsToDeblend` parent footprints with a total peak count "
            "within `ciDebledChildRange`. "
            "If `useCiLimits==False` then this parameter is ignored.")


class SourceDeblendTask(pipeBase.Task):
    """Split blended sources into individual sources.

    This task has no return value; it only modifies the SourceCatalog in-place.
    """
    ConfigClass = SourceDeblendConfig
    _DefaultName = "sourceDeblend"

    def __init__(self, schema, peakSchema=None, **kwargs):
        """Create the task, adding necessary fields to the given schema.

        Parameters
        ----------
        schema : `lsst.afw.table.Schema`
            Schema object for measurement fields; will be modified in-place.
        peakSchema : `lsst.afw.table.peakSchema`
            Schema of Footprint Peaks that will be passed to the deblender.
            Any fields beyond the PeakTable minimal schema will be transferred
            to the main source Schema. If None, no fields will be transferred
            from the Peaks
        **kwargs
            Additional keyword arguments passed to ~lsst.pipe.base.task
        """
        pipeBase.Task.__init__(self, **kwargs)
        self.schema = schema
        self.toCopyFromParent = [item.key for item in self.schema
                                 if item.field.getName().startswith("merge_footprint")]
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
        self.psfFluxKey = schema.addField('deblend_psf_instFlux', type='D',
                                          doc='If deblended-as-psf, the instrumental PSF flux', units='count')
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
        self.peakCenter = afwTable.Point2IKey.addFields(schema, name="deblend_peak_center",
                                                        doc="Center used to apply constraints in scarlet",
                                                        unit="pixel")
        self.peakIdKey = schema.addField("deblend_peakId", type=np.int32,
                                         doc="ID of the peak in the parent footprint. "
                                             "This is not unique, but the combination of 'parent'"
                                             "and 'peakId' should be for all child sources. "
                                             "Top level blends with no parents have 'peakId=0'")
        self.nPeaksKey = schema.addField("deblend_nPeaks", type=np.int32,
                                         doc="Number of initial peaks in the blend. "
                                             "This includes peaks that may have been culled "
                                             "during deblending or failed to deblend")
        self.parentNPeaksKey = schema.addField("deblend_parentNPeaks", type=np.int32,
                                               doc="Same as deblend_n_peaks, but the number of peaks "
                                                   "in the parent footprint")

    @pipeBase.timeMethod
    def run(self, exposure, sources):
        """Get the PSF from the provided exposure and then run deblend.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to be processed
        sources : `lsst.afw.table.SourceCatalog`
            SourceCatalog containing sources detected on this exposure.
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
        """Deblend.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to be processed
        srcs : `lsst.afw.table.SourceCatalog`
            SourceCatalog containing sources detected on this exposure
        psf : `lsst.afw.detection.Psf`
            Point source function

        Returns
        -------
        None
        """
        # Cull footprints if required by ci
        if self.config.useCiLimits:
            self.log.info(f"Using CI catalog limits, "
                          f"the original number of sources to deblend was {len(srcs)}.")
            # Select parents with a number of children in the range
            # config.ciDeblendChildRange
            minChildren, maxChildren = self.config.ciDeblendChildRange
            nPeaks = np.array([len(src.getFootprint().peaks) for src in srcs])
            childrenInRange = np.where((nPeaks >= minChildren) & (nPeaks <= maxChildren))[0]
            if len(childrenInRange) < self.config.ciNumParentsToDeblend:
                raise ValueError("Fewer than ciNumParentsToDeblend children were contained in the range "
                                 "indicated by ciDeblendChildRange. Adjust this range to include more "
                                 "parents.")
            # Keep all of the isolated parents and the first
            # `ciNumParentsToDeblend` children
            parents = nPeaks == 1
            children = np.zeros((len(srcs),), dtype=bool)
            children[childrenInRange[:self.config.ciNumParentsToDeblend]] = True
            srcs = srcs[parents | children]
            # We need to update the IdFactory, otherwise the the source ids
            # will not be sequential
            idFactory = srcs.getIdFactory()
            maxId = np.max(srcs["id"])
            idFactory.notify(maxId)

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
            # t0 = time.clock()

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
                self.log.warn('Parent %i: skipping large footprint (area: %i)',
                              int(src.getId()), int(fp.getArea()))
                continue
            if self.isMasked(fp, exposure.getMaskedImage().getMask()):
                src.set(self.maskedKey, True)
                self.skipParent(src, mi.getMask())
                self.log.warn('Parent %i: skipping masked footprint (area: %i)',
                              int(src.getId()), int(fp.getArea()))
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
                for key in self.toCopyFromParent:
                    child.set(key, src.get(key))
                child.assign(heavy.getPeaks()[0], self.peakSchemaMapper)
                child.setParent(src.getId())
                child.setFootprint(heavy)
                child.set(self.psfKey, peak.deblendedAsPsf)
                child.set(self.hasStrayFluxKey, peak.strayFlux is not None)
                if peak.deblendedAsPsf:
                    (cx, cy) = peak.psfFitCenter
                    child.set(self.psfCenterKey, geom.Point2D(cx, cy))
                    child.set(self.psfFluxKey, peak.psfFitFlux)
                child.set(self.deblendRampedTemplateKey, peak.hasRampedTemplate)
                child.set(self.deblendPatchedTemplateKey, peak.patched)

                # Set the position of the peak from the parent footprint
                # This will make it easier to match the same source across
                # deblenders and across observations, where the peak
                # position is unlikely to change unless enough time passes
                # for a source to move on the sky.
                child.set(self.peakCenter, geom.Point2I(pks[j].getIx(), pks[j].getIy()))
                child.set(self.peakIdKey, pks[j].getId())

                # The children have a single peak
                child.set(self.nPeaksKey, 1)
                # Set the number of peaks in the parent
                child.set(self.parentNPeaksKey, len(pks))

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
            # print('Deblending parent id', src.getId(), 'took', time.clock() - t0)

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
        """Returns whether the footprint violates the mask limits
        """
        size = float(footprint.getArea())
        for maskName, limit in self.config.maskLimits.items():
            maskVal = mask.getPlaneBitMask(maskName)
            unmaskedSpan = footprint.spans.intersectNot(mask, maskVal)  # spanset of unmasked pixels
            if (size - unmaskedSpan.getArea())/size > limit:
                return True
        return False

    def skipParent(self, source, mask):
        """Indicate that the parent source is not being deblended

        We set the appropriate flags and mask.

        Parameters
        ----------
        source : `lsst.afw.table.SourceRecord`
            The source to flag as skipped
        mask : `lsst.afw.image.Mask`
            The mask to update
        """
        fp = source.getFootprint()
        source.set(self.deblendSkippedKey, True)
        if self.config.notDeblendedMask:
            mask.addMaskPlane(self.config.notDeblendedMask)
            fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))

        # Set the center of the parent
        bbox = fp.getBBox()
        centerX = int(bbox.getMinX()+bbox.getWidth()/2)
        centerY = int(bbox.getMinY()+bbox.getHeight()/2)
        source.set(self.peakCenter, geom.Point2I(centerX, centerY))
        # There are no deblended children, so nChild = 0
        source.set(self.nChildKey, 0)
        # But we also want to know how many peaks that we would have
        # deblended if the parent wasn't skipped.
        source.set(self.nPeaksKey, len(fp.peaks))
        # Top level parents are not a detected peak, so they have no peakId
        source.set(self.peakIdKey, 0)
        # Top level parents also have no parentNPeaks
        source.set(self.parentNPeaksKey, 0)
