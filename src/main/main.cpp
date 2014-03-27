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

    const int workingWidth = originalWidth * scaleFactor;
    const int workingHeight = originalHeight * scaleFactor;

    const Eigen::Vector2d imageCenter(workingWidth / 2.0, workingHeight / 2.0);
    const double imageSize = max(workingWidth, workingHeight);

    printf("Image size = %d x %d\n", workingWidth, workingHeight);

    initImg->resize(workingWidth, workingHeight, 1, 1, 5);

    const int numPoints = 10000;

    CVOpticalFlow klt(31, 3);

    klt.init(*initImg, numPoints, min(workingWidth, workingHeight) * 0.01);

    printf("Feature count = %d\n", klt.featureCount());

    DepthReconstruction reconstruct;

    CVFundamentalMatrixEstimator fundMatEst;

    reconstruct.init(imageCount - 1, klt.featureCount());

    for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
        Eigen::Vector2f match0;
        Eigen::Vector2f matchOther;
        float error;

        klt.getMatch(pointI, match0, matchOther, error);

        match0 -= imageCenter.cast<float>();
        match0 /= imageSize;

        reconstruct.setMainPoint(pointI, match0.cast<double>());
    }

    vector<int> matchCount(klt.featureCount());

    std::fill(matchCount.begin(), matchCount.end(), 0);

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

        // Prune KLT matches with the epipolar constraint by estimating the
        // fundamental matrix and rejecting outliers.
        // Note that this actually estimates a scaled version of the essential
        // matrix.
        fundMatEst.init(klt, imageCenter[0], imageCenter[1], imageSize);

        Eigen::Matrix3d F;

        fundMatEst.estimateFundamentalMatrix(F);

        // TODO
        // Use F (which is actually the essential matrix) to compute the
        // camera matrix for the "other" image.
        //
        // Compute all 4 possible camera matrices, and choose the one
        // which results in positive z-values in both cameras, after triangulation.
        //
        
        /*
        auto FTkern = F.transpose().fullPivLu().kernel();
        Eigen::Vector3d P2e = FTkern.col(0);

        Eigen::Matrix3d ex;
        ex <<
            0, -P2e.z(), P2e.y(),
            P2e.z(), 0, -P2e.x(),
            -P2e.y(), P2e.x(), 0;
        Eigen::Matrix3d P2rotation = -ex * F;

        P2rotation += P2e * Eigen::Vector3d(1, 0, 0).transpose();
        P2rotation += P2e * Eigen::Vector3d(0, 1, 0).transpose();
        P2rotation += P2e * Eigen::Vector3d(0, 0, 1).transpose();

        Eigen::Quaterniond P2rotationQ(P2rotation);

        Eigen::Vector3d P2center = -P2rotation.transpose() * P2e;

        reconstruct.setCamera(imgI - 1, P2rotationQ, P2center);
        cout << "Camera:\n\tforward: " << P2rotation.row(0) << endl;
        cout << "\tup: " << P2rotation.row(1) << endl;
        cout << "\tcenter: " << P2center.transpose() << endl;

        Eigen::Matrix<double, 3, 4> cam0 = Eigen::Matrix<double, 3, 4>::Identity();

        Eigen::Matrix<double, 3, 4> cam1;

        cam1 <<
            P2rotation(0, 0), P2rotation(0, 1), P2rotation(0, 2), P2e(0),
            P2rotation(1, 0), P2rotation(1, 1), P2rotation(1, 2), P2e(1),
            P2rotation(2, 0), P2rotation(2, 1), P2rotation(2, 2), P2e(2);
        */

        /*
        PolarFundamentalMatrix polarF;
        bool polarFInit = false;

        for (int pointI = 0; pointI < fundMatEst.getMatchCount(); pointI++) {
            array<Eigen::Vector2d, 2> match;

            if (fundMatEst.getMatch(pointI, match[0], match[1])) {
                polarF.init(F, match);
                polarFInit = true;
                break;
            }
        }
        */

        // For visualizing matches
        /*
        CImg<uint8_t> mainMatchVis = *initImg;
        CImg<uint8_t> otherMatchVis = *curImg;
        mainMatchVis.resize(-100, -100, -100, 3);
        otherMatchVis.resize(-100, -100, -100, 3);
        CImg<uint8_t> colors = CImg<uint8_t>::lines_LUT256();

        CImg<float> depthVis(workingWidth, workingHeight);
        depthVis.fill(0);
        float totalDisp = 0;
        */

        int successfulMatches = 0;

        for (int pointI = 0; pointI < fundMatEst.getMatchCount(); pointI++) {
            Eigen::Vector2d match0;
            Eigen::Vector2d matchOther;

            if (fundMatEst.getMatch(pointI, match0, matchOther)) {
                // If this is the first time we've seen this point, then
                // triangulate it and the resulting depth as an initial
                // estimate.
                if (matchCount[pointI] == 0) {
                    // TODO compute and set depth based on triangulation

                    /*
                    triangulate(cam0, cam1, match0, matchOther, tri);

                    printf("Reconstruction: (%f, %f) -> (%f, %f, %f, %f) -> (%f, %f)\n",
                            match0.x(), match0.y(),
                            tri.x(), tri.y(), tri.z(), tri.w(),
                            (tri.x() / tri.w()) / (tri.z() / tri.w()),
                            (tri.y() / tri.w()) / (tri.z() / tri.w()));

                    reconstruct.setDepth(pointI, tri.z() / tri.w());
                    */
                }

                matchCount[pointI]++;
                // printf("Match = (%f, %f) -> (%f, %f)\n", match0[0], match0[1], matchOther[0], matchOther[1]);

                // visualize matches
                /*{
                    uint8_t color[3];

                    color[0] = colors(pointI % 256, 0);
                    color[1] = colors(pointI % 256, 1);
                    color[2] = colors(pointI % 256, 2);

                    Eigen::Vector2d match0Screen = match0 * imageSize + imageCenter;
                    Eigen::Vector2d matchOtherScreen = matchOther * imageSize + imageCenter;

                    mainMatchVis.draw_circle(match0Screen.x() + 0.5, match0Screen.y() + 0.5, 3, color);
                    otherMatchVis.draw_circle(matchOtherScreen.x() + 0.5, matchOtherScreen.y() + 0.5, 3, color);

                    double epipoleDistance0 = polarF.getEpipolarDistance(0, match0);
                    double epipoleDistance1 = polarF.getEpipolarDistance(1, matchOther);
                    float disparity = 1.0 / (epipoleDistance1 - epipoleDistance0);

                    totalDisp += disparity;

                    depthVis.draw_circle(match0Screen.x() + 0.5, match0Screen.y() + 0.5, 3, &disparity);
                }*/

                successfulMatches++;

                // fundmatEst does this normalization already
                // matchOther -= Eigen::Vector2f(workingWidth / 2.0, workingHeight / 2.0);
                // matchOther /= max(workingWidth / 2.0, workingHeight / 2.0);

                reconstruct.addObservation(imgI - 1, pointI, matchOther);
            }
        }

        printf("Found %d final matches from %d initial klt matches\n",
                successfulMatches, fundMatEst.getMatchCount());

        // (mainMatchVis, otherMatchVis).display();
        // float avgDisp = totalDisp / successfulMatches;
        // depthVis -= (depthVis.get_sign().abs() - 1) * avgDisp;
        // depthVis.display();

        /*
        if (imgI == 2) {
            reconstruct.solve(false);
        } 

        if (imgI >= 2) {
            reconstruct.solve(true);
        }
        */
    }


    vector<Eigen::Vector3d> reconstruction = reconstruct.getPoints();

    printf("[\n");
    for (int pointI = 0; pointI < reconstruction.size(); pointI++) {
        Eigen::Vector3d& p = reconstruction[pointI];

        p.x() *= max(workingWidth / 2.0, workingHeight / 2.0);
        p.y() *= max(workingWidth / 2.0, workingHeight / 2.0);
        p += Eigen::Vector3d(workingWidth / 2.0, workingHeight / 2.0, 0);

        if (matchCount[pointI] >= 3 && p.z() > 0) {
            printf("(%f, %f, %f),\n", p[0], p[1], p[2]);
        }
    }

    printf("]\n");
    
    // Sort reconstructed depth samples from back to front
    std::sort(reconstruction.begin(), reconstruction.end(), [](
                const Eigen::Vector3d& a,
                const Eigen::Vector3d& b) {
            return a[2] < b[2];
            });
    
    const Eigen::Vector3d& medianDepth = reconstruction[reconstruction.size() / 2];

    CImg<float> depthMap(workingWidth, workingHeight, 1, 1);

    depthMap  = medianDepth.z();

    for (int pointI = 0; pointI < reconstruction.size(); pointI++) {
        const Eigen::Vector3d& sample = reconstruction[pointI];

        if (matchCount[pointI] > 0) {
            depthMap.draw_circle(sample.x() + 0.5, sample.y() + 0.5, 10, &sample.z());
        }
    }

    // depthMap.blur(10);

    depthMap.display();

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
