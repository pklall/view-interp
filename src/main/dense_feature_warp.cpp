#include "dense_feature_warp.h"

#include "daisy/daisy.h"

void DenseFeatureMatch::match(
        const CImg<float>& imgGray) {

    // If these are modified, then the KDTree size MUST be changed from 200
    // to the appropriate descirptor size!
    const int rad   = 15;
    const int radq  =  3;
    const int thq   =  8;
    const int histq =  8;

    featureDetector.verbose(0);

    featureDetector.set_image(imgGray.data(), imgGray.height(),
            imgGray.width());

    featureDetector.set_parameters(rad, radq, thq, histq);

    featureDetector.set_normalization(NRM_PARTIAL);

    printf("Initializing single descriptor mode\n");
    featureDetector.initialize_single_descriptor_mode();

    printf("Creating KD Tree\n");

    const int maxLeaf = 10;

    kdtree.reset(new DaisyKDTree(200, {this},
            nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf)));

    printf("Building index\n");

    printf("Done\n");
}

