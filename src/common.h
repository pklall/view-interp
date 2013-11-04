#pragma once

#include <vector>
#include <string>
#include <functional>
#include <limits>
#include "stdio.h"

#include <limits>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#define cimg_use_png
#define cimg_use_jpeg
#define cimg_use_jpeg
#include "CImg.h"

using namespace std;

using namespace cimg_library;

extern template struct cimg_library::CImg<float>;

inline double norm(float x, float y) {
    return sqrt(x * x + y * y);
}
