#include "common.h"

#include "cvutil/cvutil.h"

// #include "dense_feature_warp.h"

#include "polar_stereo.h"

#include "klt_feature_warp.h"

#include "reconstruct.h"

#include <Eigen/Dense>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s img1.png img2.png ... imgN.png\n", argv[0]);
        exit(1);
    }

    const int imageCount = argc - 1;

    unique_ptr<CImg<uint8_t>> initImg(new CImg<uint8_t>());
    unique_ptr<CImg<uint8_t>> curImg(new CImg<uint8_t>());

    // Load a grayscale image from RGB
    *initImg = CImg<float>::get_load(argv[1]);
    if (initImg->spectrum() > 1) {
        *initImg = initImg->get_RGBtoLab().channel(0);
    }

    int originalWidth = initImg->width();
    int originalHeight = initImg->height();

    // More manageable size
    float scaleFactor = 2.0f * 1000000.0f / ((float) originalWidth * originalHeight);

    // Don't increase image size
    scaleFactor = min(1.0f, scaleFactor);

    int workingWidth = originalWidth * scaleFactor;
    int workingHeight = originalHeight * scaleFactor;

    printf("Image size = %d x %d\n", workingWidth, workingHeight);

    initImg->resize(workingWidth, workingHeight, 1, 1, 5);

    const int numPoints = 10000;

    CVOpticalFlow klt(31, 3);

    klt.init(*initImg, numPoints, min(workingWidth, workingHeight) * 0.01);

    printf("Feature count = %d\n", klt.featureCount());

    Reconstruction reconstruct;

    reconstruct.init(imageCount, klt.featureCount(), 1.0);

    for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
        Eigen::Vector2f match0;
        Eigen::Vector2f matchOther;
        float error;

        klt.getMatch(pointI, match0, matchOther, error);

        match0 -= Eigen::Vector2f(workingWidth / 2.0, workingHeight / 2.0);

        match0.x() /= max(workingWidth, workingHeight);
        match0.y() /= max(workingWidth, workingHeight);

        reconstruct.addObservation(0, pointI, match0);
    }

    for (int imgI = 1; imgI < imageCount; imgI++) {
        printf("Processing image #%d\n", imgI);

        *curImg = CImg<float>::get_load(argv[1 + imgI]);
        if (curImg->spectrum() > 1) {
            *curImg = curImg->get_RGBtoLab().channel(0);
        }
        assert(curImg->width() == originalWidth);
        assert(curImg->height() == originalHeight);
        curImg->resize(workingWidth, workingHeight, 1, 1, 5);

        klt.compute(*curImg);

        for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
            Eigen::Vector2f match0;
            Eigen::Vector2f matchOther;
            float error;

            if (klt.getMatch(pointI, match0, matchOther, error)) {
                matchOther -= Eigen::Vector2f(workingWidth / 2.0, workingHeight / 2.0);

                matchOther.x() /= max(workingWidth, workingHeight);
                matchOther.y() /= max(workingWidth, workingHeight);

                reconstruct.addObservation(imgI, pointI, matchOther);
            }
        }
    }

    reconstruct.solve();

    CImg<uint8_t> depthMap(workingWidth, workingHeight, 1, 1, 5);

    vector<Eigen::Vector3d> reconstruction(klt.featureCount());

    printf("[\n");
    for (int i = 0; i < klt.featureCount(); i++) {
        Eigen::Vector3d p = reconstruct.getCameraSpacePoint(0, i);
        
        p.x() *= max(workingWidth, workingHeight);
        p.y() *= max(workingWidth, workingHeight);

        p.x() += workingWidth / 2.0;
        p.y() += workingHeight / 2.0;

        printf("(%f, %f, %f),\n", p[0], p[1], p[2]);
        
        reconstruction[i] = p;
    }

    printf("]\n");
    
    // Sort reconstructed depth samples from back to front
    std::sort(reconstruction.begin(), reconstruction.end(), [](
                const Eigen::Vector3d& a,
                const Eigen::Vector3d& b) {
            return a[2] < b[2];
            });
    
    const Eigen::Vector3d& medianDepth = reconstruction[reconstruction.size() / 2];
    
    return 0;
}

/*
   int main(int argc, char** argv) {
   if (argc < 3) {
   printf("Usage: %s img1.png img2.png ... imgN.png\n", argv[0]);
   exit(1);
   }

   const int imageCount = argc - 1;

   unique_ptr<CImg<uint8_t>> initImg(new CImg<uint8_t>());
   unique_ptr<CImg<uint8_t>> curImg(new CImg<uint8_t>());

   const int maxFeatures = 8096;

// Larger patch size is necessary for high-resolution images.
// Note that detecting features on full-size images is ideal for greatest
// precision in computing the fundamental matrix.
const int patchSize = 31;

unique_ptr<CVFeatureMatcher> prevFeat(
new CVFeatureMatcher(maxFeatures, patchSize));
unique_ptr<CVFeatureMatcher> curFeat(
new CVFeatureMatcher(maxFeatures, patchSize));

CVFundamentalMatrixEstimator fEstimator;

// Load a grayscale image from RGB
 *initImg = CImg<uint8_t>::get_load(argv[1]).get_RGBtoLab().channel(0);

 int originalWidth = initImg->width();
 int originalHeight = initImg->height();

// More manageable size of 1 megapixel
float scaleFactor = 1.0f * 1000000.0f / (originalWidth * originalHeight);

int workingWidth = originalWidth * scaleFactor;
int workingHeight = originalHeight * scaleFactor;

prevFeat->detectFeatures(*initImg);

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
 fEstimator.init(*prevFeat, *curFeat);
 fEstimator.estimateFundamentalMatrix(fundMat);
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

initImg->resize(workingWidth, workingHeight, 1, 1, 5);
curImg->resize(workingWidth, workingHeight, 1, 1, 5);

F.scale(originalWidth, originalHeight, workingWidth, workingHeight);

printf("Initializing rectification...\n");
rectification.init(curImg->width(), curImg->height(), F);
printf("Done\n");

// Multiple scales, downsampling by 0.75 each time
stereo.computeStereo(1, 0.75f, F, *initImg, *curImg);

const auto& disp = stereo.getDisparityAtScale(0);

disp.display();

// disp.get_equalize(255).get_map(CImg<float>::cube_LUT256()).display();

// (((disp - disp.median()) / (1.96f * pow(disp.variance(3), 2.0f))) * 127.0f + 127.0f).display();//.get_map(CImg<float>::cube_LUT256()).display();
}

return 0;
}
*/

/*
   int main(int argc, char** argv) {
   if (argc < 3) {
   printf("Usage: %s img1.png img2.png ... imgN.png\n", argv[0]);
   exit(1);
   }

   const int imageCount = argc - 1;

   unique_ptr<CImg<uint8_t>> prevImg(new CImg<uint8_t>());
   unique_ptr<CImg<uint8_t>> curImg(new CImg<uint8_t>());

// Load a grayscale image from RGB
 *prevImg = CImg<float>::get_load(argv[1]).get_RGBtoLab().channel(0);

 int originalWidth = prevImg->width();
 int originalHeight = prevImg->height();

// More manageable size
float scaleFactor = 0.5f * 1000000.0f / (originalWidth * originalHeight);

int workingWidth = originalWidth * scaleFactor;
int workingHeight = originalHeight * scaleFactor;

PolarStereo stereo;

PolarRectification rectification;

for (int imgI = 1; imgI < imageCount; imgI++) {
printf("Processing image #%d\n", imgI);

 *curImg = CImg<float>::get_load(argv[1 + imgI]).get_RGBtoLab().channel(0);
 assert(curImg->width() == originalWidth);
 assert(curImg->height() == originalHeight);

 prevImg->resize(workingWidth, workingHeight, 1, 1, 5);
 curImg->resize(workingWidth, workingHeight, 1, 1, 5);

 DenseFeatureMatch match;

 match.match(*prevImg);

 swap(prevImg, curImg);
 }

 return 0;
 }
 */
