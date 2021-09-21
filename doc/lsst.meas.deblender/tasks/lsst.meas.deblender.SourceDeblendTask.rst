.. lsst-task-topic:: lsst.meas.deblender.SourceDeblendTask

#################
SourceDeblendTask
#################

``SourceDeblendTask`` splits blended sources into individual sources.

.. _lsst.meas.deblender.SourceDeblendTask-summary:

Processing summary
==================

``SourceDeblendTask`` performs housekeeping work — updating table
schemas and extracting the PSF from the input image — then delegates
to `lsst.meas.deblender.deblend` to perform deblending.
The task has no return value; the input
`~lsst.afw.table.SourceCatalog` is modified in-place.

.. _lsst.meas.deblender.SourceDeblendTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.meas.deblender.SourceDeblendTask

.. _lsst.meas.deblender.SourceDeblendTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.meas.deblender.SourceDeblendTask

.. _lsst.meas.deblender.SourceDeblendTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.meas.deblender.SourceDeblendTask

.. _lsst.meas.deblender.SourceDeblendTask-examples:
