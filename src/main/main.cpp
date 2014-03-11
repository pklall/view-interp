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

    const int maxFeatures = 512;
    const int patchSize = 31;

    unique_ptr<CVFeatureMatcher> prevFeat(
            new CVFeatureMatcher(maxFeatures, patchSize));
    unique_ptr<CVFeatureMatcher> curFeat(
            new CVFeatureMatcher(maxFeatures, patchSize));

    CVFundamentalMatrixEstimator fEstimator;

    PolarRectification rectification;

    prevImg->load(argv[1]);

    prevFeat->detectFeatures(prevImg->get_RGBtoLab().channel(0));

    for (int imgI = 1; imgI < imageCount; imgI++) {
        printf("Processing image #%d\n", imgI);

        curImg->load(argv[1 + imgI]);

        curFeat->detectFeatures(curImg->get_RGBtoLab().channel(0));

        Eigen::Matrix3f F;

        fEstimator.estimateFundamentalMatrix(*prevFeat, *curFeat, F);

        array<Eigen::Vector2f, 2> match;

        int numMatches = fEstimator.getMatchCount();

        printf("Match count = %d\n", numMatches);

        for (int i = 0; i < numMatches; i++) {
            if (fEstimator.getMatch(i, match[0], match[1])) {
                break;
            }
        }

        // TODO get match from estimator

        rectification.init(curImg->width(), curImg->height(), F, match);

        swap(prevImg, curImg);
        swap(prevFeat, curFeat);
    }

    return 0;
}

