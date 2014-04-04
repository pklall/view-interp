#include "common.h"

#include "cvutil/cvutil.h"

// #include "dense_feature_warp.h"

#include "polar_stereo.h"

#include "klt_feature_warp.h"

#include "depth_reconstruction.h"

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
    *initImg = CImg<uint8_t>::get_load(argv[1]);
    if (initImg->spectrum() > 1) {
        *initImg = initImg->get_RGBtoLab().channel(0);
    }

    int originalWidth = initImg->width();
    int originalHeight = initImg->height();

    const int workingWidth = originalWidth;
    const int workingHeight = originalHeight;

    const Eigen::Vector2d imageCenter(workingWidth / 2.0, workingHeight / 2.0);
    const double imageSize = max(workingWidth / 2.0, workingHeight / 2.0);

    printf("Image size = %d x %d\n", workingWidth, workingHeight);

    const int numPoints = 10000;

    // CVFeatureMatcher orb(numPoints, 127);
    CVOpticalFlow klt(31, 10);

    // orb.detectFeatures(*initImg);
    klt.init(*initImg, numPoints, min(workingWidth, workingHeight) * 0.01);

    printf("Feature count = %d\n", klt.featureCount());

    DepthReconstruction reconstruct;

    CVFundamentalMatrixEstimator fundMatEst;

    // reconstruct.init(imageCount - 1, orb.numKeypoints());
    reconstruct.init(imageCount - 1, klt.featureCount());

    // for (int pointI = 0; pointI < orb.numKeypoints(); pointI++) {
    for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
        Eigen::Vector2f match0;
        Eigen::Vector2f matchOther;
        float error;

        klt.getMatch(pointI, match0, matchOther, error);
        // float x, y;
        // orb.getKeypoint(pointI, x, y);

        // Eigen::Vector2f match0(x, y);
        
        match0 -= imageCenter.cast<float>();
        match0 /= imageSize;

        reconstruct.setKeypoint(pointI, match0.cast<double>());
    }


    // CVFeatureMatcher curOrb(numPoints, 31);
    // vector<tuple<int, int>> matches(numPoints);

    for (int imgI = 1; imgI < imageCount; imgI++) {
        printf("Processing image #%d\n", imgI);

        *curImg = CImg<uint8_t>::get_load(argv[1 + imgI]);

        if (curImg->spectrum() > 1) {
            *curImg = curImg->get_RGBtoLab().channel(0);
        }

        assert(curImg->width() == originalWidth);
        assert(curImg->height() == originalHeight);

        printf("Computing KLT\n");
        klt.compute(*curImg);
        // curOrb.detectFeatures(*curImg);
        // matches.clear();
        // orb.match(curOrb, matches, numPoints);
        printf("Done\n");

        // for (const tuple<int, int> match : matches) {
        for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
            Eigen::Vector2f match0;
            Eigen::Vector2f matchOther;
            float error;

            klt.getMatch(pointI, match0, matchOther, error);

            // float x, y;
            // curOrb.getKeypoint(get<1>(match), x, y);
            // int pointI = get<0>(match);
            // Eigen::Vector2f matchOther(x, y);

            matchOther -= imageCenter.cast<float>();
            matchOther /= imageSize;

            reconstruct.addObservation(imgI - 1, pointI, matchOther.cast<double>());
        }
    }

    reconstruct.solve();

    // Visualize the result
    vector<tuple<Eigen::Vector2d, vector<double>>> depthSamples;

    reconstruct.getAllDepthSamples(depthSamples);

    CImg<double> depthVis(workingWidth, workingHeight);

    depthVis = 0.0f;

    double totalDepth = 0.0;
    size_t numDepth = 0;

    for (auto& pointDepthSamples : depthSamples) {
        Eigen::Vector2d point = get<0>(pointDepthSamples);
        vector<double>& depths = get<1>(pointDepthSamples);
        
        if (depths.size() == 0) {
            continue;
        }

        point *= max(depthVis.width(), depthVis.height()) / 2.0;
        point.x() += depthVis.width() / 2.0;
        point.y() += depthVis.height() / 2.0;

        // TODO use nth element instead
        std::sort(depths.begin(), depths.end());

        double d = depths[depths.size() / 2];

        totalDepth += d;
        numDepth++;

        depthVis.draw_circle(point.x() + 0.5, point.y() + 0.5, 5, &d);
    }

    float avgDepth = totalDepth / numDepth;
    depthVis -= (depthVis.get_sign().abs() - 1) * avgDepth;

    depthVis.display();

    const bool display_rectification = false;

    if (display_rectification) {
        CImg<uint8_t> initDown(*initImg);
        CImg<uint8_t> curDown;

        float scaleFactor = (1024.0f * 1024.0f) / ((float) originalWidth * originalHeight);
        int scaledWidth = scaleFactor * originalWidth;
        int scaledHeight = scaleFactor * originalHeight; 

        // Resize with moving-average interpolation
        initDown.resize(scaledWidth, scaledHeight, -100, -100, 2);

        for (int imgI = 1; imgI < imageCount; imgI++) {
            printf("Rectifying image #%d\n", imgI);

            curDown = CImg<uint8_t>::get_load(argv[1 + imgI]);

            if (curDown.spectrum() > 1) {
                curDown = curDown.get_RGBtoLab().channel(0);
            }

            curDown.resize(scaledWidth, scaledHeight, -100, -100, 2);

            PolarFundamentalMatrix polarF;
            PolarRectification polarR;
            PolarStereo polarS;

            bool rectificationPossible = reconstruct.getPolarFundamentalMatrix(
                    imgI - 1,
                    Eigen::Vector2d(scaledWidth / 2.0, scaledHeight / 2.0),
                    max(scaledWidth / 2.0, scaledHeight / 2.0),
                    polarF);

            if (!rectificationPossible) {
                printf("Rectification not possible, epipoles at infinity.\n");
                continue;
            }

            // Resize to a workable size and adjust the fundamental matrix
            // accordingly.

            printf("Initializing rectification...\n");
            polarR.init(curDown.width(), curDown.height(), polarF);
            printf("Done\n");

            // Multiple scales, downsampling by 0.75 each time
            CImg<uint8_t> rectified0;
            CImg<uint8_t> rectified1;
            CImg<float> revMap;
            polarR.rectify(0, initDown, rectified0, revMap);
            polarR.rectify(1, curDown, rectified1, revMap);

            rectified0.save(("./results/rectified_" + to_string(imgI) + "_l.png").c_str());
            rectified1.save(("./results/rectified_" + to_string(imgI) + "_r.png").c_str());
            // (rectified0, rectified1).display();

            // polarS.computeStereo(1, 0.75f, polarF, initDown, curDown);
            // const auto& disp = polarS.getDisparityAtScale(0);
            // disp.display();
        }
    }

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
