import math
import os
import pylab as plt
import numpy as np
from matplotlib.patches import Ellipse

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable


# To use multiprocessing, we need the plot elements to be picklable.  Swig objects are not
# picklable, so in preprocessing we pull out the items we need for plotting, putting them in
# a _MockSource object.

class _MockSource:

    def __init__(self, src, mi, psfkey, fluxkey, xkey, ykey, flagKeys, ellipses=True,
                 maskbit=None):
        # flagKeys: list of (key, string) tuples
        self.sid = src.getId()
        aa = {}
        if maskbit is not None:
            aa.update(mask=True)
        self.im = footprintToImage(src.getFootprint(), mi, **aa).getArray()
        if maskbit is not None:
            self.im = ((self.im & maskbit) > 0)

        self.x0 = mi.getX0()
        self.y0 = mi.getY0()
        self.ext = getExtent(src.getFootprint().getBBox())
        self.ispsf = src.get(psfkey)
        self.psfflux = src.get(fluxkey)
        self.flags = [nm for key, nm in flagKeys if src.get(key)]
        # self.cxy = (src.get(xkey), src.get(ykey))
        self.cx = src.get(xkey)
        self.cy = src.get(ykey)
        pks = src.getFootprint().getPeaks()
        self.pix = [pk.getIx() for pk in pks]
        self.piy = [pk.getIy() for pk in pks]
        self.pfx = [pk.getFx() for pk in pks]
        self.pfy = [pk.getFy() for pk in pks]
        if ellipses:
            self.ell = (src.getX(), src.getY(), src.getIxx(), src.getIyy(), src.getIxy())
    # for getEllipses()

    def getX(self):
        return self.ell[0] + 0.5

    def getY(self):
        return self.ell[1] + 0.5

    def getIxx(self):
        return self.ell[2]

    def getIyy(self):
        return self.ell[3]

    def getIxy(self):
        return self.ell[4]


def plotDeblendFamily(*args, **kwargs):
    X = plotDeblendFamilyPre(*args, **kwargs)
    plotDeblendFamilyReal(*X, **kwargs)

# Preprocessing: returns _MockSources for the parent and kids


def plotDeblendFamilyPre(mi, parent, kids, dkids, srcs, sigma1, ellipses=True, maskbit=None, **kwargs):
    schema = srcs.getSchema()
    psfkey = schema.find("deblend_deblendedAsPsf").key
    fluxkey = schema.find('deblend_psfFlux').key
    xkey = schema.find('base_NaiveCentroid_x').key
    ykey = schema.find('base_Naivecentroid_y').key
    flagKeys = [(schema.find(keynm).key, nm)
                for nm, keynm in [('EDGE', 'base_PixelFlags_flag_edge'),
                                  ('INTERP', 'base_PixelFlags_flag_interpolated'),
                                  ('INT-C', 'base_PixelFlags_flag_interpolatedCenter'),
                                  ('SAT', 'base_PixelFlags_flag_saturated'),
                                  ('SAT-C', 'base_PixelFlags_flag_saturatedCenter'),
                                  ]]
    p = _MockSource(parent, mi, psfkey, fluxkey, xkey, ykey, flagKeys, ellipses=ellipses, maskbit=maskbit)
    ch = [_MockSource(kid, mi, psfkey, fluxkey, xkey, ykey, flagKeys,
                      ellipses=ellipses, maskbit=maskbit) for kid in kids]
    dch = [_MockSource(kid, mi, psfkey, fluxkey, xkey, ykey, flagKeys,
                       ellipses=ellipses, maskbit=maskbit) for kid in dkids]
    return (p, ch, dch, sigma1)

# Real thing: make plots given the _MockSources


def plotDeblendFamilyReal(parent, kids, dkids, sigma1, plotb=False, idmask=None, ellipses=True,
                          arcsinh=True, maskbit=None):
    if idmask is None:
        idmask = ~0
    pim = parent.im
    pext = parent.ext

    N = 1 + len(kids)
    S = math.ceil(math.sqrt(N))
    C = S
    R = math.ceil(float(N) / C)

    def nlmap(X):
        return np.arcsinh(X / (3.*sigma1))

    def myimshow(im, **kwargs):
        arcsinh = kwargs.pop('arcsinh', True)
        if arcsinh:
            kwargs = kwargs.copy()
            mn = kwargs.get('vmin', -5*sigma1)
            kwargs['vmin'] = nlmap(mn)
            mx = kwargs.get('vmax', 100*sigma1)
            kwargs['vmax'] = nlmap(mx)
            plt.imshow(nlmap(im), **kwargs)
        else:
            plt.imshow(im, **kwargs)

    imargs = dict(interpolation='nearest', origin='lower',
                  vmax=pim.max(), arcsinh=arcsinh)
    if maskbit:
        imargs.update(vmin=0)

    plt.figure(figsize=(8, 8))
    plt.clf()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9,
                        wspace=0.05, hspace=0.1)
    plt.subplot(R, C, 1)
    myimshow(pim, extent=pext, **imargs)
    plt.gray()
    plt.xticks([])
    plt.yticks([])
    m = 0.25
    pax = [pext[0]-m, pext[1]+m, pext[2]-m, pext[3]+m]
    x, y = parent.pix[0], parent.piy[0]
    tt = 'parent %i @ (%i,%i)' % (parent.sid & idmask,
                                  x - parent.x0, y - parent.y0)
    if len(parent.flags):
        tt += ', ' + ', '.join(parent.flags)
    plt.title(tt)
    Rx, Ry = [], []
    tts = []
    stys = []
    xys = []
    for i, kid in enumerate(kids):
        ext = kid.ext
        plt.subplot(R, C, i+2)
        if plotb:
            ima = imargs.copy()
            ima.update(vmax=max(3.*sigma1, kid.im.max()))
        else:
            ima = imargs

        myimshow(kid.im, extent=ext, **ima)
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        tt = 'child %i' % (kid.sid & idmask)
        if kid.ispsf:
            sty1 = dict(color='g')
            sty2 = dict(color=(0.1, 0.5, 0.1), lw=2, alpha=0.5)
            tt += ' (psf: flux %.1f)' % kid.psfflux
        else:
            sty1 = dict(color='r')
            sty2 = dict(color=(0.8, 0.1, 0.1), lw=2, alpha=0.5)

        if len(kid.flags):
            tt += ', ' + ', '.join(kid.flags)

        tts.append(tt)
        stys.append(sty1)
        plt.title(tt)
        # bounding box
        xx = [ext[0], ext[1], ext[1], ext[0], ext[0]]
        yy = [ext[2], ext[2], ext[3], ext[3], ext[2]]
        plt.plot(xx, yy, '-', **sty1)
        Rx.append(xx)
        Ry.append(yy)
        # peak(s)
        plt.plot(kid.pfx, kid.pfy, 'x', **sty2)
        xys.append((kid.pfx, kid.pfy, sty2))
        # centroid
        plt.plot([kid.cx], [kid.cy], 'x', **sty1)
        xys.append(([kid.cx], [kid.cy], sty1))
        # ellipse
        if ellipses and not kid.ispsf:
            drawEllipses(kid, ec=sty1['color'], fc='none', alpha=0.7)
        if plotb:
            plt.axis(ext)
        else:
            plt.axis(pax)

    # Go back to the parent plot and add child bboxes
    plt.subplot(R, C, 1)
    for rx, ry, sty in zip(Rx, Ry, stys):
        plt.plot(rx, ry, '-', **sty)
    # add child centers and ellipses...
    for x, y, sty in xys:
        plt.plot(x, y, 'x', **sty)
    if ellipses:
        for kid, sty in zip(kids, stys):
            if kid.ispsf:
                continue
            drawEllipses(kid, ec=sty['color'], fc='none', alpha=0.7)
    plt.plot([parent.cx], [parent.cy], 'x', color='b')
    if ellipses:
        drawEllipses(parent, ec='b', fc='none', alpha=0.7)

    # Plot dropped kids
    for kid in dkids:
        ext = kid.ext
        # bounding box
        xx = [ext[0], ext[1], ext[1], ext[0], ext[0]]
        yy = [ext[2], ext[2], ext[3], ext[3], ext[2]]
        plt.plot(xx, yy, 'y-')
        # peak(s)
        plt.plot(kid.pfx, kid.pfy, 'yx')
    plt.axis(pax)


def footprintToImage(fp, mi=None, mask=False):
    if not fp.isHeavy():
        fp = afwDet.makeHeavyFootprint(fp, mi)
    bb = fp.getBBox()
    if mask:
        im = afwImage.MaskedImageF(bb.getWidth(), bb.getHeight())
    else:
        im = afwImage.ImageF(bb.getWidth(), bb.getHeight())
    im.setXY0(bb.getMinX(), bb.getMinY())
    fp.insert(im)
    if mask:
        im = im.getMask()
    return im


def getFamilies(cat):
    '''
    Returns [ (parent0, children0), (parent1, children1), ...]
    '''
    # parent -> [children] map.
    children = {}
    for src in cat:
        pid = src.getParent()
        if not pid:
            continue
        if pid in children:
            children[pid].append(src)
        else:
            children[pid] = [src]
    keys = sorted(children.keys())
    return [(cat.find(pid), children[pid]) for pid in keys]


def getExtent(bb, addHigh=1):
    # so verbose...
    return (bb.getMinX(), bb.getMaxX()+addHigh, bb.getMinY(), bb.getMaxY()+addHigh)


def cutCatalog(cat, ndeblends, keepids=None, keepxys=None):
    fams = getFamilies(cat)
    if keepids:
        # print 'Keeping ids:', keepids
        # print 'parent ids:', [p.getId() for p,kids in fams]
        fams = [(p, kids) for (p, kids) in fams if p.getId() in keepids]
    if keepxys:
        keep = []
        pts = [afwGeom.Point2I(x, y) for x, y in keepxys]
        for p, kids in fams:
            for pt in pts:
                if p.getFootprint().contains(pt):
                    keep.append((p, kids))
                    break
        fams = keep

    if ndeblends:
        # We want to select the first "ndeblends" parents and all their children.
        fams = fams[:ndeblends]

    keepcat = afwTable.SourceCatalog(cat.getTable())
    for p, kids in fams:
        keepcat.append(p)
        for k in kids:
            keepcat.append(k)
    keepcat.sort()
    return keepcat


def readCatalog(sourcefn, heavypat, ndeblends=0, dataref=None,
                keepids=None, keepxys=None,
                patargs=dict()):
    if sourcefn is None:
        cat = dataref.get('src')
        try:
            if not cat:
                return None
        except Exception:
            return None
    else:
        if not os.path.exists(sourcefn):
            print('No source catalog:', sourcefn)
            return None
        print('Reading catalog:', sourcefn)
        cat = afwTable.SourceCatalog.readFits(sourcefn)
        print(len(cat), 'sources')
    cat.sort()
    cat.defineCentroid('base_SdssCentroid')

    if ndeblends or keepids or keepxys:
        cat = cutCatalog(cat, ndeblends, keepids, keepxys)
        print('Cut to', len(cat), 'sources')

    if heavypat is not None:
        print('Reading heavyFootprints...')
        for src in cat:
            if not src.getParent():
                continue
            dd = patargs.copy()
            dd.update(id=src.getId())
            heavyfn = heavypat % dd
            if not os.path.exists(heavyfn):
                print('No heavy footprint:', heavyfn)
                return None
            mim = afwImage.MaskedImageF(heavyfn)
            heavy = afwDet.makeHeavyFootprint(src.getFootprint(), mim)
            src.setFootprint(heavy)
    return cat


def datarefToMapper(dr):
    return dr.butlerSubset.butler.mapper


def datarefToButler(dr):
    return dr.butlerSubset.butler


class WrapperMapper:

    def __init__(self, real):
        self.real = real
        for x in dir(real):
            if not x.startswith('bypass_'):
                continue

            class RelayBypass:

                def __init__(self, real, attr):
                    self.func = getattr(real, attr)
                    self.attr = attr

                def __call__(self, *args):
                    # print('relaying', self.attr)
                    # print('to', self.func)
                    return self.func(*args)
            setattr(self, x, RelayBypass(self.real, x))
            # print('Wrapping', x)

    def map(self, *args, **kwargs):
        print('Mapping', args, kwargs)
        R = self.real.map(*args, **kwargs)
        print('->', R)
        return R
    # relay

    def isAggregate(self, *args):
        return self.real.isAggregate(*args)

    def getKeys(self, *args):
        return self.real.getKeys(*args)

    def getDatasetTypes(self):
        return self.real.getDatasetTypes()

    def queryMetadata(self, *args):
        return self.real.queryMetadata(*args)

    def canStandardize(self, *args):
        return self.real.canStandardize(*args)

    def standardize(self, *args):
        return self.real.standardize(*args)

    def validate(self, *args):
        return self.real.validate(*args)

    def getDefaultLevel(self, *args):
        return self.real.getDefaultLevel(*args)


def getEllipses(src, nsigs=[1.], **kwargs):
    xc = src.getX()
    yc = src.getY()
    x2 = src.getIxx()
    y2 = src.getIyy()
    xy = src.getIxy()
    # SExtractor manual v2.5, pg 29.
    a2 = (x2 + y2)/2. + np.sqrt(((x2 - y2)/2.)**2 + xy**2)
    b2 = (x2 + y2)/2. - np.sqrt(((x2 - y2)/2.)**2 + xy**2)
    theta = np.rad2deg(np.arctan2(2.*xy, (x2 - y2)) / 2.)
    a = np.sqrt(a2)
    b = np.sqrt(b2)
    ells = []
    for nsig in nsigs:
        ells.append(Ellipse([xc, yc], 2.*a*nsig, 2.*b*nsig, angle=theta, **kwargs))
    return ells


def drawEllipses(src, **kwargs):
    els = getEllipses(src, **kwargs)
    for el in els:
        plt.gca().add_artist(el)
    return els


def get_sigma1(mi):
    stats = afwMath.makeStatistics(mi.getVariance(), mi.getMask(), afwMath.MEDIAN)
    sigma1 = math.sqrt(stats.getValue(afwMath.MEDIAN))
    return sigma1
