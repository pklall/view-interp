#include "CImg.h"

using namespace cimg_library;

// Compile CImg specialized for these types.
namespace cimg_library {
    template struct CImg<float>;
    template struct CImg<double>;
    template struct CImg<int>;
    template struct CImg<bool>;
}
