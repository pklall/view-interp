#include "common.h"

#include "util.h"

#include "pmstereo/pmstereo.h"

#include "cvstereo/cvstereo.h"

using namespace std;

using namespace cimg_library;

#define dbgOut std::cout

#define NUM_SAMPLES 14
string SAMPLES[NUM_SAMPLES][2] = {
    // Not rectified:
    // {"samples/football0.jpg", "samples/football1.jpg"},
    // {"samples/bed0.jpg", "samples/bed1.jpg"},
    // Middleburry rectified:
    {"samples/chess0.png", "samples/chess1.png"},
    {"samples/art0.png", "samples/art1.png"},
    {"samples/computer0.png", "samples/computer1.png"},
    {"samples/laundry0.png", "samples/laundry1.png"},
    {"samples/cones0.png", "samples/cones1.png"},
    {"samples/newspaper0.ppm", "samples/newspaper1.ppm"},
    {"samples/books0.png", "samples/books1.png"},
    {"samples/bull0.ppm", "samples/bull1.ppm"},
    {"samples/drumsticks0.png", "samples/drumsticks1.png"},
    {"samples/dwarves0.png", "samples/dwarves1.png"},
    {"samples/hall0.jpg", "samples/hall1.jpg"},
    {"samples/moebius0.png", "samples/moebius1.png"},
    {"samples/poster0.ppm", "samples/poster1.ppm"},
    {"samples/raindeer0.png", "samples/raindeer1.png"}
};

int main(int argc, char** argv) {
    for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES; sampleIndex++) {
        CImg<float> fst(SAMPLES[sampleIndex][0].c_str());
        CImg<float> lst(SAMPLES[sampleIndex][1].c_str());

        CVStereo stereo(fst, lst, true);

        CImg<float> rLeft, rRight;
        stereo.getRectified(rLeft, rRight);
        // CImgList<float>(rLeft, rRight).display();

        stereo.matchStereo();

        CImg<float> result;

        stereo.getStereo(result);

        result.display();
    }

    return 1;
}

void testPMStereo(
        CImg<float>& fst,
        CImg<float>& lst) {
    int MAX_SIZE = 1024;

    while (fst.width() > MAX_SIZE || fst.height() > MAX_SIZE) {
        fst.resize_halfXY();
        lst.resize_halfXY();
    }

    // The number of "particles" to use
    int K = 1;

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
    int wndSize = 31;
    int iterations = 3;

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

    CImg<bool> consistentLeft(fieldLeftSlice.width(), fieldLeftSlice.height());
    translationalConsistency(fieldLeftSlice, fieldRightSlice, consistentLeft);

    CImg<float> visLeft = fieldLeftSlice;

    cimg_forXY(visLeft, x, y) {
        if (!consistentLeft(x, y)) {
            cimg_forZC(visLeft, z, c) {
                visLeft(x, y, z, c) = 0.0f;
            }
        }
    }

    cimg_forXY(visLeft, x, y) {
        if (!consistentLeft(x, y)) {
            cimg_forZC(visLeft, z, c) {
                visLeft(x, y, z, c) = 0.0f;
            }
        }
    }

    CImgList<float>(fst, visLeft).display();
}


