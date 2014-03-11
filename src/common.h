#pragma once

#include <vector>
#include <iterator>
#include <string>
#include <functional>
#include <limits>
#include <cstdio>
#include <memory>
#include <cassert>
#include <iostream>

#define cimg_use_png
#define cimg_use_jpeg
#include "cimg/CImg.h"

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
