#
# LSST Data Management System
#
# Copyright 2018  AURA/LSST.
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
import os
import unittest

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class SchemaTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.calexp = afwImage.ExposureF(os.path.join(DATA_DIR, "ticket1738.fits"))

    def tearDown(self):
        del self.calexp

    def testMismatchedSchema(self):
        schema = afwTable.SourceTable.makeMinimalSchema()

        # Create the detection task and process the data
        detectionTask = SourceDetectionTask(schema=schema)
        table = afwTable.SourceTable.make(schema)
        result = detectionTask.run(table, self.calexp)
        self.assertEqual(schema, result.sources.getSchema())

        # SourceDeblendTask modifies the schema in-place at construction
        # and add extra keys to it
        deblendTask = SourceDeblendTask(schema)
        self.assertNotEqual(schema, result.sources.getSchema())

        # As the deblendTask has a different schema than the original schema
        # of the detectionTask, the assert should fail and stop running
        with self.assertRaises(AssertionError):
            deblendTask.run(self.calexp, result.sources)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
