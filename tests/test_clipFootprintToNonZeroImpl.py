

#!/usr/bin/env python
#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import print_function
import unittest

import lsst.utils.tests
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.meas.deblender.plugins import clipFootprintToNonzeroImpl


class ClipFootprintTestCase(lsst.utils.tests.TestCase):
    '''
    This is s a test to verify that clipping footprints to a non-zero region
    works as intended. It is expected that spans which contain leading or
    trailing zeros will be trimmed as to not include these values, while zeros
    which occur inside a span will be preserved.
    '''
    def testClip(self):
        im = afwImage.ImageI(afwGeom.Box2I(afwGeom.Point2I(-2, -2),
                                           afwGeom.Extent2I(20, 20)))

        span1 = afwGeom.SpanSet.fromShape(5, afwGeom.Stencil.BOX, (6, 6))
        span1.setImage(im, 20)

        im.getArray()[6+2, 6+2] = 0

        # Set some negative pixels to ensure they are not clipped
        im.getArray()[12,:] *= -1

        span2 = afwGeom.SpanSet.fromShape(6, afwGeom.Stencil.BOX, (6, 6))

        foot = afwDet.Footprint(span2)

        clipFootprintToNonzeroImpl(foot, im)

        self.assertEqual(foot.spans, span1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
