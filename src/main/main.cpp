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

    // Larger patch size is necessary for highe resolution images.
    // Note that detecting features on full-size images is ideal for greatest
    // precision in computing the fundamental matrix.
    const int patchSize = 128;

    unique_ptr<CVFeatureMatcher> prevFeat(
            new CVFeatureMatcher(maxFeatures, patchSize));
    unique_ptr<CVFeatureMatcher> curFeat(
            new CVFeatureMatcher(maxFeatures, patchSize));

    CVFundamentalMatrixEstimator fEstimator;

    prevImg->load(argv[1]);

    prevFeat->detectFeatures(prevImg->get_RGBtoLab().channel(0));

    PolarFundamentalMatrix F;

    PolarRectification rectification;

    for (int imgI = 1; imgI < imageCount; imgI++) {
        printf("Processing image #%d\n", imgI);

        curImg->load(argv[1 + imgI]);

        printf("Detecting features...\n");
        curFeat->detectFeatures(curImg->get_RGBtoLab().channel(0));
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

        printf("Initializing rectification...\n");
        rectification.init(curImg->width(), curImg->height(), F);
        printf("Done\n");

        CImg<uint8_t> rectified;
        CImg<float> reverseMap;

        rectification.rectify(0, *prevImg, rectified, reverseMap);
        rectified.save(("./results/rectified_" + to_string(imgI) + "_left.png").c_str());

        rectification.rectify(1, *curImg, rectified, reverseMap);

        rectified.save(("./results/rectified_" + to_string(imgI) + "_right.png").c_str());

        swap(prevImg, curImg);
        swap(prevFeat, curFeat);
    }

    return 0;
}

