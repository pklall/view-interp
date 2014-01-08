#include "common.h"

#include "util.h"

#include "files.h"

#include "pmstereo/pmstereo.h"

#include <omp.h>

using namespace std;

using namespace cimg_library;

#define dbgOut std::cout


int main(int argc, char** argv) {
    for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES; sampleIndex++) {
        CImg<float> fst(SAMPLES[sampleIndex][0].c_str());
        CImg<float> lst(SAMPLES[sampleIndex][1].c_str());

        int MAX_SIZE = 1024;

        while (fst.width() > MAX_SIZE || fst.height() > MAX_SIZE) {
            fst.resize_halfXY();
            lst.resize_halfXY();
        }

        // The number of "particles" to use
        int K = 5;

        CImg<float> labLeft = fst.get_RGBtoLab();
        CImg<float> labRight  = lst.get_RGBtoLab();

        CImg<float> gradLeft;
        CImg<float> gradRight;
        {
            CImgList<float> gradLeftLst = labLeft.get_shared_channel(0).get_gradient();
            gradLeft = (gradLeftLst(0).sqr() + gradLeftLst(1).sqr()).sqrt();

            CImgList<float> gradRightLst = labRight.get_shared_channel(0).get_gradient();
            gradRight = (gradRightLst(0).sqr() + gradRightLst(1).sqr()).sqrt();
        }

        CImg<float> fieldLeft(labLeft.width(), labLeft.height(), K, 1);
        CImg<float> fieldRight(labRight.width(), labRight.height(), K, 1);
        // Initialize with random, (mostly) valid disparity particles
        cimg_forXYZ(fieldLeft, x, y, z) {
            int randX = (int) (cimg::rand() * labRight.width());

            fieldLeft(x, y, z) = randX - x;
        }
        cimg_forXYZ(fieldRight, x, y, z) {
            int randX = (int) (cimg::rand() * labLeft.width());

            fieldLeft(x, y, z) = randX - x;
        }

        CImg<int> sortedLeft(labLeft.width(), labLeft.height(), K);
        CImg<int> sortedRight(labRight.width(), labRight.height(), K);
        cimg_forXYZ(sortedLeft, x, y, z) {
            sortedLeft(x, y, z) = z;
        }
        cimg_forXYZ(sortedRight, x, y, z) {
            sortedRight(x, y, z) = z;
        }

        CImg<float> distLeft(labLeft.width(), labLeft.height(), K);
        CImg<float> distRight(labRight.width(), labRight.height(), K);

        distLeft = std::numeric_limits<float>::max();
        distRight = std::numeric_limits<float>::max();

        // Used to store slices of the final result
        CImg<float> fieldLeftSlice(fieldLeft.width(), fieldLeft.height());
        CImg<float> fieldRightSlice(fieldRight.width(), fieldRight.height());

        float randomSearchFactor = 1.0f;
        int increment = 1;
        int wndSize = 11;
        int iterations = 5;

        patchMatchTranslationalCorrespondence(
                labLeft,    labRight,
                gradLeft,   gradRight,
                fieldLeft,  fieldRight,
                distLeft,   distRight,
                sortedLeft, sortedRight,
                wndSize,
                iterations,
                randomSearchFactor,
                increment);

        // Save the best result
        cimg_forXY(fieldLeftSlice, x, y) {
            fieldLeftSlice(x, y) =
                fieldLeft(x, y, sortedLeft(0));
        }

        cimg_forXY(fieldRightSlice, x, y) {
            fieldRightSlice(x, y) =
                fieldRight(x, y, sortedRight(0));
        }

        fieldLeftSlice.display();
    }

    return 1;
}


