#pragma once

#include <vector>
#include <iterator>
#include <string>
#include <functional>
#include <limits>
#include <cstdio>

#include <cassert>

// OpenGM and Halide MUST be included before CImg, which includes
// X11 headers with conflicting/stupid definitions (i.e. `#define BOOL`).
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"
#include "opengm/inference/messagepassing/messagepassing_bp.hxx"
#include "opengm/inference/messagepassing/messagepassing_trbp.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/mqpbo.hxx"
#include "opengm/inference/graphcut.hxx"
#include "opengm/inference/alphaexpansion.hxx"
#include "opengm/inference/alphabetaswap.hxx"
#include "opengm/inference/auxiliary/minstcutboost.hxx"

#include "Halide.h"


#define cimg_use_png
#define cimg_use_jpeg
#define cimg_use_jpeg
#include "cimg/CImg.h"

#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

using namespace cimg_library;

#ifdef PRECOMPILE_CIMG

extern template struct cimg_library::CImg<float>;
extern template struct cimg_library::CImg<double>;
extern template struct cimg_library::CImg<int>;
extern template struct cimg_library::CImg<bool>;

#endif

inline double norm(float x, float y) {
    return sqrt(x * x + y * y);
}

template<class T>
inline T sqr(T v) {
    return v * v;
}
