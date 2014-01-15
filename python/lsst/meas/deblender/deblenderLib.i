// -*- lsst-c++ -*-
%define meas_deblenderLib_DOCSTRING
"
Python interface to lsst::meas::deblender classes
"
%enddef

%feature("autodoc", "1");

%module(package="lsst.meas.deblender", docstring=meas_deblenderLib_DOCSTRING) deblenderLib

%{
#include <exception>
#include <list>
#include <boost/shared_ptr.hpp>
#include "lsst/meas/deblender/Baseline.h"
#include "lsst/afw/table.h"
#include "lsst/afw/detection.h"
#include "lsst/pex/logging.h"
#include "lsst/afw/cameraGeom.h"
#include "lsst/afw/math.h"

#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/ImageSlice.h"
#include "lsst/afw/image/TanWcs.h"
%}

%inline %{
namespace lsst { namespace afw {
        namespace detection { }
        namespace image { }
} }
using namespace lsst;
using namespace lsst::afw::image;
using namespace lsst::afw::detection;
%}

%include "lsst/p_lsstSwig.i"
%include "lsst/base.h"                  // PTR(); should be in p_lsstSwig.i

%lsst_exceptions();

%import "lsst/afw/table/io/ioLib.i"
%import "lsst/afw/image/imageLib.i"
%import "lsst/afw/detection/detectionLib.i"

%apply bool *OUTPUT { bool *patchedEdges };

%include "lsst/meas/deblender/Baseline.h"

%extend lsst::meas::deblender::BaselineUtils<T> {
  %template(copyWithinFootprintImage)       copyWithinFootprint<lsst::afw::image::Image<T> >;
  %template(copyWithinFootprintMaskedImage) copyWithinFootprint<lsst::afw::image::MaskedImage<T> >;
};

/*
%extend lsst::meas::deblender::BaselineUtils {
    %template(copyWithinFootprintImageF)       copyWithinFootprint< lsst::afw::image::Image<float> >;
    %template(copyWithinFootprintMaskedImageF) copyWithinFootprint< lsst::afw::image::MaskedImage<float> >;
 };
 */

%template(BaselineUtilsF) lsst::meas::deblender::BaselineUtils<float>;

//%template(BaselineUtilsFcopyWithinFootprintImage) lsst::meas::deblender::BaselineUtils<float>::copyWithinFootprint<lsst::afw::image::Image<float> >;
//%template(BaselineUtilsFcopyWithinFootprintMaskedImage) lsst::meas::deblender::BaselineUtils<float>::copyWithinFootprint<lsst::afw::image::MaskedImage<float> >;


%template(pairMaskedImageFPtrAndFootprintPtr) std::pair<lsst::meas::deblender::BaselineUtils<float>::MaskedImagePtrT, lsst::meas::deblender::BaselineUtils<float>::FootprintPtrT>;

%template(HeavyFootprintPtrListF) std::vector<boost::shared_ptr<lsst::afw::detection::HeavyFootprint<float, lsst::afw::image::MaskPixel,lsst::afw::image::VariancePixel> > >;

// HeavyFootprintPtr* is instantiated in afw/det/footprints.i

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
