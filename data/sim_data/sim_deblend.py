#!/usr/bin/env python
import galsim
import os
import numpy as np
import pylab
import scipy.spatial
import argparse

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.coord as afwCoord
import lsst.meas.algorithms as measAlg
import lsst.afw.detection as afwDet
import lsst.afw.display.ds9 as ds9
import lsst.meas.modelfit
import lsst.meas.extensions.convolved

from lsst.meas.deblender.baseline import deblend,PerPeak
from lsst.meas.algorithms import  SourceDetectionTask, SingleFrameMeasurementTask
from lsst.meas.deblender import SourceDeblendTask


def matchCatalogs(x_ref, y_ref, x_p, y_p):
    posRef = np.dstack([x_ref, y_ref])[0]
    posP = np.dstack([x_p, y_p])[0]
    mytree = scipy.spatial.cKDTree(posRef)
    dist, index = mytree.query(posP)

    return dist, index

class powerLaw:

    def __init__(self,min, max, gamma, rng):
        self.min=min
        self.max=max
        self.gamma_p1= gamma+1
        self.rng = rng
        if self.gamma_p1 == 0:
            self.base = np.log(self.min)
            self.norm = np.log(self.max/self.min)
        else:
            self.base = np.power(self.min, self.gamma_p1);
            self.norm = np.power(self.max, self.gamma_p1) - self.base;

    def sample(self):
        v = self.rng() * self.norm + self.base;
        if self.gamma_p1 == 0:
            return np.exp(v)
        else:
             return np.power(v, 1./self.gamma_p1);

class TrueGal:

    def __init(self):
        self.x = 0
        self.y = 0
        self.rad = 0
        self.sed = 0
        self.redshift = 0
        self.fluxes = {}



parser = argparse.ArgumentParser(description='Run image with multiple filters')
parser.add_argument('--seed', type=int, default=0, help='random see')
parser.add_argument('--max_gal', type=int, default=40, help='maximum number of galaxies/cluster ')
parser.add_argument('--tile', type=int, default=3, help='number of clusters in single direction per image')
parser.add_argument('--image_sizex', type=int, default=300, help='size of single cluster in x')
parser.add_argument('--image_sizey', type=int, default=300, help='size of single cluster in y')
parser.add_argument('--min_dist', type=int, default=15, help='minimum distance of new galaxy from existing galaxies')
parser.add_argument('--filters', default='ugrizy', help='filters to use only allows ugrizy')
parser.add_argument('--filter_norm', default='i', help='filter to use as normalization')
parser.add_argument('--max_flux', type=float, default=500, help='maximum flux')
parser.add_argument('--min_flux', type=float, default=1, help='minimum flux')
parser.add_argument('--slope_flux', type=float, default=-0.75, help='power law slope of flux')
parser.add_argument('--max_size', type=float, default=10, help='maximum size')
parser.add_argument('--min_size', type=float, default=0.1, help='minimum size')
parser.add_argument('--slope_size', type=float, default=-1, help='power law slope of size')
parser.add_argument('--min_pos', type=float, default=10, help='minimum position from center of image')
parser.add_argument('--slope_pos', type=float, default=-0.3, help='power law slope of position')
parser.add_argument('--psf_size', type=float, default=1.9, help='size of gaussian psf')
parser.add_argument('--noise_sigma', type=float, default=0.1, help='sigma of gaussian noise in image')
parser.add_argument('--max_redshift', type=float, default=1.5, help='maximum redsfhit')
parser.add_argument('--output_dir', default='./', help='output directory')
parser.add_argument('--n_templates', type=int, default=100, help='number of different sed templates')
parser.add_argument('--grow', type=int, default=7, help='grow footprints by this*psf')

args = parser.parse_args()
seed = args.seed
rng = galsim.UniformDeviate(seed)
np.random.seed(seed)
max_gal = args.max_gal
tile = args.tile
image_sizey = args.image_sizey
image_sizex = args.image_sizex
scale = 1.
psf_size = args.psf_size
noise_sigma = args.noise_sigma
min_dist = args.min_dist
filter_names = args.filters
flux_pl = powerLaw(args.min_flux, args.max_flux, args.slope_flux, rng)
size_pl = powerLaw(args.min_size, args.max_size, args.slope_size, rng)
size_pl = powerLaw(0.1, 10, -1, rng)
# keep objects away from edges
pos_pl = powerLaw(args.min_pos, 0.5*np.sqrt((image_sizey/2)**2+(image_sizex/2)**2), args.slope_pos, rng)
max_redshift = args.max_redshift
match_radius = 0.5
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

noise = galsim.GaussianNoise(rng, sigma=noise_sigma)
psf = galsim.Gaussian(sigma=psf_size)
psf = psf.shear(e1=0.03, e2=0.05)

true_gals = []

sed_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
seds = []
for sed_name in sed_names:
        sed_filename = os.path.join(galsim.meta_data.share_dir, '{0}.sed'.format(sed_name))
        sed = galsim.SED(sed_filename, wave_type='Ang', flux_type='flambda')
        seds.append(sed.withFluxDensity(target_flux_density=1.0, wavelength=500))

n_templates = args.n_templates
templates = np.random.rand(n_templates, len(sed_names))
# normalize contribution from each template
templates = [ a/np.sum(a) for a in templates]
template_seds = []
for template in templates:
    sed = template[0]*seds[0]
    for v,t in zip(template[1:],seds[1:]):
        sed += v*t
    template_seds.append(sed)

filters = {}
filter_path = '/tigress/rea3/hsc_sim/GalSim/examples/data/'
for filter_name in filter_names:
        filter_filename = os.path.join(filter_path, 'LSST_{0}.dat'.format(filter_name))
        filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

tot_bounds = galsim.BoundsI(0,image_sizex*tile, 0, image_sizey*tile)
tot_images = {}
for filter_name in filters.keys():
    tot_images[filter_name] = galsim.ImageF(tot_bounds, scale=scale)


true_x = []
true_y = []
current_image = 0
for itile in range(tile):
    for jtile in range(tile):
        n_gal = int(np.random.rand()*max_gal)
        bounds = galsim.BoundsI(itile*image_sizex,(itile+1)*image_sizex, jtile*image_sizey, (jtile+1)*image_sizey)
        center = bounds.center()
        images = {}
        for filter_name in filters.keys():
            images[filter_name] = galsim.ImageF(bounds, scale=scale)

        print 'generating image %d/%d with %d galaxies' % (current_image,tile*tile, n_gal)
        current_image += 1
        for i in range(n_gal):

            # Make sure next galaxy is suffiently away from already inserted galaxies
            while True:
                r = pos_pl.sample()
                theta = np.random.rand()*2*np.pi
                offsetx = r*np.cos(theta)
                offsety = r*np.sin(theta)

                gtrue = TrueGal()
                gtrue.fluxes = {}
                gtrue.x = offsetx + center.x
                gtrue.y = offsety + center.y

                dist, index = matchCatalogs([gtrue.x], [gtrue.y], true_x, true_y)
                if np.sum(dist < min_dist) == 0 :
                        break

            exp_e1 = 2
            exp_e2 = 2
            while np.sqrt(exp_e1**2 + exp_e2**2) > 0.7:
                exp_e1 = np.random.randn()*0.3
                exp_e2 = np.random.randn()*0.3
            exp_radius = size_pl.sample()

            dev_e1 = 2
            dev_e2 = 2
            while np.sqrt(dev_e1**2 + dev_e2**2) > 0.7:
                dev_e1 = np.random.randn()*0.3
                dev_e2 = np.random.randn()*0.3
            dev_radius = size_pl.sample()

            disk = galsim.Exponential(half_light_radius=exp_radius*scale)
            disk = disk.shear(e1=exp_e1, e2=exp_e2)

            frac = np.random.rand()

            bulge_frac = frac
            bulge = galsim.DeVaucouleurs(half_light_radius=dev_radius*scale)
            bulge = bulge.shear(e1=dev_e2, e2=dev_e2)

            flux = flux_pl.sample()
            gal = frac*bulge + (1-frac)*disk

            redshift = np.random.rand()*max_redshift
            sed = np.random.choice(template_seds,1)[0].atRedshift(redshift)

            #normalize flux according to specified filter
            sed = sed.withFlux(flux, filters[args.filter_norm])
            mgal = galsim.Chromatic(gal, sed)

            cgal = galsim.Convolve([mgal, psf])

            gtrue.rad = frac*dev_radius+(1-frac)*exp_radius
            gtrue.sed = sed
            gtrue.redshift = redshift

            true_x.append(gtrue.x)
            true_y.append(gtrue.y)

            for filter_name,filter_ in filters.items():
                tmp = cgal.drawImage(filter_, image=images[filter_name], add_to_image=True, offset=(offsetx,offsety))
                gtrue.fluxes[filter_name] = tmp.added_flux

            true_gals.append(gtrue)

        for filter_name,filter_ in filters.items():
            tot_images[filter_name][bounds] = images[filter_name]


# Write out truth catalogs
true_schema = afwTable.Schema()
xKey = true_schema.addField("x",type=float,doc="x position")
yKey = true_schema.addField("y",type=float,doc="y position")
radKey = true_schema.addField("size",type=float,doc="size")
zKey = true_schema.addField("redshift",type=float,doc="redshift")
fluxKeys = {}
for filter_ in filter_names:
    fluxKeys[filter_] = true_schema.addField("flux_%s" % filter_, type=float, doc="flux in %s" % filter_)
w_vals = np.arange(100, 1500, 5)
wKey = true_schema.addField("wave", type="ArrayF", doc="wavelengths", size=len(w_vals))
sedKey = true_schema.addField("sed", type="ArrayF", doc="sed at wavelengths", size=len(w_vals))

true_cat = afwTable.BaseCatalog(true_schema)
for gal in true_gals:
    rec= true_cat.addNew()
    rec.set(xKey, gal.x)
    rec.set(yKey, gal.y)
    rec.set(radKey, gal.rad)
    rec.set(zKey, gal.redshift)
    rec.set(wKey, w_vals.astype(np.float32))
    sed_vals = np.array([gal.sed(a) for a in w_vals])
    rec.set(sedKey, sed_vals.astype(np.float32))
    for filter_ in filter_names:
        rec.set(fluxKeys[filter_], gal.fluxes[filter_])
true_cat.writeFits('%s/catalog_true.fits' % args.output_dir)

# add Noise and write image/psf
for filter_name,image in images.items():
    tot_images[filter_name].addNoise(noise)
    tot_images[filter_name].write('%s/image_%s.fits' % (output_dir, filter_name))

psf_bounds = galsim.BoundsI(0, 41, 0, 41)
psf_image = galsim.ImageF(psf_bounds, scale=scale)
psf.drawImage(image=psf_image)
psf_image.write('%s/psf_image.fits' % output_dir)

# Read in PSF
lsst_psf_image = afwImage.ImageF('%s/psf_image.fits' % output_dir)
bbox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(41, 41))
lsst_psf_image = lsst_psf_image[bbox].convertD()
lsst_psf_image = afwMath.offsetImage(lsst_psf_image, -0.5, -0.5)
kernel = afwMath.FixedKernel(lsst_psf_image)
kernelPsf = measAlg.KernelPsf(kernel)

# Exposure setup
wcs = afwImage.makeWcs(
    afwCoord.Coord(0.*afwGeom.degrees, 0.*afwGeom.degrees),
    afwGeom.Point2D(0.0,0.0), 0.168*3600, 0.0, 0.0, 0.168*3600
)
calib = afwImage.Calib()
calib.setFluxMag0(1e12)

# Config and task setup
measure_config = SingleFrameMeasurementTask.ConfigClass()
measure_config.load('/tigress/rea3/lsst/DM-8059/obs_subaru/config/apertures.py')
measure_config.load('/tigress/rea3/lsst/DM-8059/obs_subaru/config/kron.py')

measure_config.plugins.names |= ["modelfit_DoubleShapeletPsfApprox", "modelfit_CModel"]
measure_config.slots.modelFlux = 'modelfit_CModel'
algName = "base_CircularApertureFlux"
algName2 = "ext_convolved_ConvolvedFlux"
for subConfig in [measure_config.undeblended]:
        subConfig.names = [algName]
        subConfig[algName].radii = [3,6,12,24]
        subConfig[algName].maxSincRadius = 0  # Disable sinc photometry because we're undersampled

for subConfig in (measure_config.plugins, measure_config.undeblended):
    subConfig.names.add(algName2)
    values = [ii/0.168 for ii in (0.65, 0.85, 1.1)]
    algConfig = subConfig[algName2]
    algConfig.seeing = values
    algConfig.aperture.radii = [3,6,12,24]
    algConfig.aperture.maxSincRadius = 0

measure_config.load('/tigress/rea3/lsst/DM-8059/obs_subaru/config/apertures.py')

deblend_config = SourceDeblendTask.ConfigClass()
detect_config = SourceDetectionTask.ConfigClass()
detect_config.isotropicGrow = True
detect_config.doTempLocalBackground = False
detect_config.thresholdValue = 5
detect_config.nSigmaToGrow = args.grow

schema = afwTable.SourceTable.makeMinimalSchema()
detectTask = SourceDetectionTask(schema, config=detect_config)
deblendTask = SourceDeblendTask(schema, config=deblend_config)
measureTask = SingleFrameMeasurementTask(schema, config=measure_config)

peakMinimalSchema = afwDet.PeakTable.makeMinimalSchema()
peakSchemaMapper = afwTable.SchemaMapper(peakMinimalSchema, schema)

truthKey = schema.addField('truth_index', type=int, doc='Index to truth catalog')

table = afwTable.SourceTable.make(schema)

lsst_images = {}
for filter_ in filter_names:
    lsst_images[filter_] = afwImage.ImageF('%s/image_%s.fits' % (output_dir, filter_))

# Run only detection on each filter
detections = {}
exposures = {}
for filter_ in filter_names:
    print 'detecting in ', filter_
    lsst_image = lsst_images[filter_]

    exposure = afwImage.ExposureF(lsst_image.getBBox(afwImage.PARENT))
    exposure.getMaskedImage().getImage().getArray()[:,:] = lsst_image.getArray()

    exposure.getMaskedImage().getVariance().set(noise_sigma**2)
    exposure.setPsf(kernelPsf)
    exposure.setWcs(wcs)
    exposure.setCalib(calib)
    exposures[filter_] = exposure
    result = detectTask.makeSourceCatalog(table, exposure)
    detections[filter_] = result.sources
    result.sources.writeFits('%s/det_%s.fits' % (args.output_dir, filter_))

# merge detection lists together
merged = afwDet.FootprintMergeList(schema, [filter_ for filter_ in filter_names])

id_factory = afwTable.IdFactory.makeSource(0, 32)
merged_sources = merged.getMergedSourceCatalog([detections[filter_] for filter_ in filter_names],
                                            [filter_ for filter_ in filter_names], 6,
                                            schema, id_factory, 1.8)
for record in merged_sources:
    record.getFootprint().sortPeaks()

print 'Total merged objects', len(merged_sources)
merged_sources.writeFits('%s/det_merge.fits' % (args.output_dir))

catalogs = {}
for filter_ in filter_names:
    print 'deblend, measure', filter_
    exposure = exposures[filter_]
    fwhm = exposure.getPsf().computeShape().getDeterminantRadius()*2.35
    sources = afwTable.SourceCatalog(merged_sources)

    for ii,src in enumerate(sources):
        deb = deblend(src.getFootprint(), exposure.getMaskedImage(), exposure.getPsf(), fwhm, verbose=False,
                      weightTemplates=False,
                      maxNumberOfPeaks=0,
                      rampFluxAtEdge=True,
                      assignStrayFlux=True, strayFluxAssignment='trim', strayFluxToPointSources='necessary',
                      findStrayFlux=True, clipStrayFluxFraction=0.001,
                      psfChisqCut1=1.5, psfChisqCut2=1.5,
                      monotonicTemplate=True,
                      medianSmoothTemplate=True, medianFilterHalfsize=2,
                      tinyFootprintSize=2,
                      clipFootprintToNonzero=True
        )

        parent = src
        parent.assign(deb.peaks[0].peak, peakSchemaMapper)
        parent.setParent(0)
        parent.setFootprint(src.getFootprint())

        for j, peak in enumerate(deb.peaks):
            heavy = peak.getFluxPortion()
            child = sources.addNew()
            child.assign(heavy.getPeaks()[0], peakSchemaMapper)
            child.setParent(1)
            child.setFootprint(heavy)
            #fig=pylab.figure()
            #ax=fig.add_subplot(111)
            #ax.imshow(peak.templateImage.getArray(),interpolation="None",vmax=1.4,cmap='gray')
            #fig.savefig('temp_%d_%d.png'%(ii,j))

    measureTask.run(sources, exposure)
    # Match to truth catalog
    for record in sources:
        if record.getParent()==0 and record.get('deblend_nChild')!=0:
            continue
        dist, index = matchCatalogs([record.getX()], [record.getY()], true_x, true_y)
        min_dist = np.min(dist)
        if min_dist < match_radius:
            nearest_index = np.where(dist==min_dist)[0][0]
            record.set(truthKey, nearest_index)

    catalogs[filter_] = sources
    sources.writeFits('%s/meas_%s.fits' % (output_dir, filter_))
    exposure.writeFits('%s/calexp_%s.fits' % (output_dir, filter_))
