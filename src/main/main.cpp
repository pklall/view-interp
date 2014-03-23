#include "common.h"

#include "cvutil/cvutil.h"

#include "polar_rectification.h"

#include <Eigen/Dense>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s img1.png img2.png ... imgN.png\n", argv[0]);
        exit(1);
    }

    const int imageCount = argc - 1;

    unique_ptr<CImg<uint8_t>> prevImg(new CImg<uint8_t>());
    unique_ptr<CImg<uint8_t>> curImg(new CImg<uint8_t>());

    const int maxFeatures = 4096;

    // Larger patch size is necessary for high-resolution images.
    // Note that detecting features on full-size images is ideal for greatest
    // precision in computing the fundamental matrix.
    const int patchSize = 63;

    unique_ptr<CVFeatureMatcher> prevFeat(
            new CVFeatureMatcher(maxFeatures, patchSize));
    unique_ptr<CVFeatureMatcher> curFeat(
            new CVFeatureMatcher(maxFeatures, patchSize));

    CVFundamentalMatrixEstimator fEstimator;

    // Load a grayscale image from RGB
    *prevImg = CImg<uint8_t>::get_load(argv[1]).get_RGBtoLab().channel(0);

    int originalWidth = prevImg->width();
    int originalHeight = prevImg->height();

    // More manageable size
    float scaleFactor = 2.0f * 1000000.0f / (originalWidth * originalHeight);

    int workingWidth = originalWidth * scaleFactor;
    int workingHeight = originalHeight * scaleFactor;

    prevFeat->detectFeatures(*prevImg);

    PolarFundamentalMatrix F;

    PolarStereo stereo;

    PolarRectification rectification;

    for (int imgI = 1; imgI < imageCount; imgI++) {
        printf("Processing image #%d\n", imgI);

        *curImg = CImg<uint8_t>::get_load(argv[1 + imgI]).get_RGBtoLab().channel(0);
        assert(curImg->width() == originalWidth);
        assert(curImg->height() == originalHeight);

        printf("Detecting features...\n");
        curFeat->detectFeatures(*curImg);
        printf("Done\n");

        printf("Estimating fundamental matrix...\n");
        Eigen::Matrix3d fundMat;
        fEstimator.estimateFundamentalMatrix(*prevFeat, *curFeat, fundMat);
        printf("Done\n");

        cout << "F = " << endl;
        cout << fundMat;
        cout << endl << endl;

        array<Eigen::Vector2d, 2> match;

        int numMatches = fEstimator.getMatchCount();

        printf("Match count = %d\n", numMatches);

        for (int i = 0; i < numMatches; i++) {
            if (fEstimator.getMatch(i, match[0], match[1])) {
                break;
            }
        }

        bool rectificationPossible = F.init(fundMat, match);

        if (!rectificationPossible) {
            printf("Rectification not possible, epipoles at infinity.\n");
            continue;
        }

        // Resize to a workable size and adjust the fundamental matrix
        // accordingly.

        prevImg->resize(workingWidth, workingHeight, 1, 1, 5);
        curImg->resize(workingWidth, workingHeight, 1, 1, 5);

        F.scale(originalWidth, originalHeight, workingWidth, workingHeight);

        printf("Initializing rectification...\n");
        rectification.init(curImg->width(), curImg->height(), F);
        printf("Done\n");


        // 4 scales, downsampling by 0.75 each time
        stereo.computeStereo(4, 0.75f, F, *prevImg, *curImg);

        /*
        CImg<uint8_t> rectified;
        CImg<float> reverseMap;

        rectification.rectify(0, *prevImg, rectified, reverseMap);
        rectified.save(("./results/rectified_" + to_string(imgI) + "_left.png").c_str());

        rectification.rectify(1, *curImg, rectified, reverseMap);

        rectified.save(("./results/rectified_" + to_string(imgI) + "_right.png").c_str());
        */

        swap(prevImg, curImg);
        swap(prevFeat, curFeat);
    }

    return 0;
}

