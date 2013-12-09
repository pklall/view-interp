#pragma once

#include <vector>
#include <string>
#include <functional>
#include <limits>
#include "stdio.h"

#include <cassert>

#include <omp.h>

#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#define cimg_use_png
#define cimg_use_jpeg
#define cimg_use_jpeg
#include "cimg/CImg.h"

using namespace std;

using namespace cimg_library;

extern template struct cimg_library::CImg<float>;
extern template struct cimg_library::CImg<int>;

inline double norm(float x, float y) {
    return sqrt(x * x + y * y);
}
