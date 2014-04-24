#include "common.h"

#include "cvutil/cvutil.h"

#include "tri_qpbo.h"

// #include "sparse_daisy_stereo.h"
// #include "sparse_interpolation.hpp"
// #include "klt_feature_warp.h"

#include "polar_stereo.h"

#include "depth_reconstruction.h"

#include <Eigen/Dense>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s img1.png img2.png ... imgN.png\n", argv[0]);
        exit(1);
    }

    const int imageCount = argc - 1;

    CImg<float> initImg;
    CImg<uint8_t> initImgGray;
    CImg<float> curImg;
    CImg<uint8_t> curImgGray;

    // Load a grayscale image from RGB
    initImg = CImg<float>::get_load(argv[1]).RGBtoLab();

    initImgGray = initImg.get_shared_channel(0);

    int originalWidth = initImg.width();
    int originalHeight = initImg.height();

    const int workingWidth = originalWidth;
    const int workingHeight = originalHeight;

    const Eigen::Vector2d imageCenter(workingWidth / 2.0, workingHeight / 2.0);
    const double imageSize = max(workingWidth / 2.0, workingHeight / 2.0);

    printf("Image size = %d x %d\n", workingWidth, workingHeight);

    const int numPoints = 20000;

    CVOpticalFlow klt(31, 15);

    float minDistance = min(workingWidth, workingHeight) * 1.0 / sqrt((float) numPoints);
    minDistance = max(5.0f, minDistance);

    klt.init(initImgGray, numPoints, minDistance);

    const int numGoodPoints = klt.sortFeatures(min(workingWidth, workingHeight) * 1.0 / sqrt(2000));

    printf("Feature count = %d\n", klt.featureCount());

    DepthReconstruction reconstruct;

    CVFundamentalMatrixEstimator fundMatEst;

    reconstruct.init(imageCount - 1, klt.featureCount(), numGoodPoints);

    vector<Eigen::Vector2f> keypoints;

    for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
        Eigen::Vector2f match0;
        Eigen::Vector2f matchOther;
        float error;

        klt.getMatch(pointI, match0, matchOther, error);

        keypoints.push_back(match0);

        match0 -= imageCenter.cast<float>();
        match0 /= imageSize;

        reconstruct.setKeypoint(pointI, match0.cast<double>());
    }

    for (int imgI = 1; imgI < imageCount; imgI++) {
        printf("Processing image #%d\n", imgI);

        curImg = CImg<float>::get_load(argv[1 + imgI]).RGBtoLab();

        curImgGray = curImg.get_shared_channel(0);

        assert(curImg.width() == originalWidth);
        assert(curImg.height() == originalHeight);

        printf("Computing KLT\n");
        klt.compute(curImgGray);
        printf("Done\n");

        for (int pointI = 0; pointI < klt.featureCount(); pointI++) {
            Eigen::Vector2f match0;
            Eigen::Vector2f matchOther;
            float error;

            klt.getMatch(pointI, match0, matchOther, error);

            matchOther -= imageCenter.cast<float>();
            matchOther /= imageSize;

            reconstruct.addObservation(imgI - 1, pointI, matchOther.cast<double>());
        }
    }

    reconstruct.solve();

    // Visualize the result
    if (true) {
        CImg<float> depthVis(workingWidth * 0.25, workingHeight * 0.25);

        reconstruct.visualize(depthVis, 1, 0.99, 1.0, false);

        depthVis.display();
    }

    {
        vector<double> depth;
        TriQPBO qpbo(initImg, keypoints);

        for (int camI = 0; camI < imageCount - 1; camI++) {
            if (camI < 0 || reconstruct.isInlierCamera(camI)) {
                reconstruct.getAllDepthSamples(camI, depth);

                qpbo.addCandidateVertexDepths(depth);

                // CImg<float> colorVis(workingWidth, workingHeight, 1, 3);
                // colorVis.fill(0);
                // qpbo.visualizeTriangulation(colorVis);
                // colorVis.display();
            }
        }

        printf("initializing qpbo\n");
        qpbo.init();
        printf("done\n");

        while(true) {
            CImg<double> depthVis(workingWidth, workingHeight);
            depthVis.fill(0.0);
            qpbo.denseInterp(depthVis);
            double medianDepth = depthVis.median();
            depthVis.min(medianDepth * 10);
            depthVis.max(0.0);
            depthVis.display();

            qpbo.solve();
        }

        /*
        for (int camI = 0; camI < imageCount - 1; camI++) {
            if (camI < 0 || reconstruct.isInlierCamera(camI)) {
                vector<double> depth; // reconstruct.getDepths();
                reconstruct.getAllDepthSamples(camI, depth);

                printf("initializing qpbo\n");
                TriQPBO qpbo(initImg, keypoints, depth);
                printf("done\n");

                   // CImg<float> colorVis(workingWidth, workingHeight, 1, 3);
                   // colorVis.fill(0);
                   // qpbo.visualizeTriangulation(colorVis);
                   // colorVis.display();

                qpbo.solve();

                CImg<double> depthVis(workingWidth, workingHeight);
                depthVis.fill(0.0);
                qpbo.denseInterp(depthVis);
                double medianDepth = depthVis.median();
                depthVis.min(medianDepth * 10);
                depthVis.max(0.0);
                depthVis.display();
            }
        }
        */

        /*
        for (int camI = -1; camI < imageCount - 1; camI++) {
            if (camI < 0 || reconstruct.isInlierCamera(camI)) {
                TriQPBO qpbo(initImg, keypoints);

                vector<double> depth(keypoints.size());

                if (camI < 0) {
                    depth = reconstruct.getDepths();
                } else {
                    reconstruct.getAllDepthSamples(camI, depth);
                }

                qpbo.addCandidateVertexDepths(depth, false);
                qpbo.solve();

                CImg<double> depthVis(workingWidth, workingHeight);
                depthVis.fill(0.0);
                qpbo.denseInterp(depthVis);
                double medianDepth = depthVis.median();
                depthVis.min(medianDepth * 10);
                depthVis.display();
            }
        }
        */
    }


#if false
    const bool display_daisy_stereo = false;

    if (display_daisy_stereo) {
        SparseDaisyStereo daisyStereo;

        daisyStereo.init(initImg.get_shared_channel(0));

        vector<Eigen::Vector2f> pointSamples;

        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                pointSamples.push_back(Eigen::Vector2f(
                            x * workingWidth / 128.0f,
                            y * workingHeight / 128.0f));
            }
        }

        for (int imgI = 1; imgI < imageCount; imgI++) {
            printf("Rectifying image #%d\n", imgI);

            curImg = CImg<float>::get_load(argv[1 + imgI]).RGBtoLab();

            PolarFundamentalMatrix polarF;

            bool rectificationPossible = reconstruct.getPolarFundamentalMatrix(
                    imgI - 1,
                    Eigen::Vector2d(workingWidth / 2.0, workingHeight / 2.0),
                    max(workingWidth / 2.0, workingHeight / 2.0),
                    polarF);

            if (!rectificationPossible) {
                printf("Rectification not possible, epipoles at infinity.\n");
                continue;
            }

            vector<Eigen::Vector2f> matches(pointSamples.size());
            vector<float> matchDistances(pointSamples.size());

            daisyStereo.match(curImg.get_shared_channel(0), polarF, pointSamples, matches,
                    matchDistances);
        }
    }
#endif

#if false
    bool display_dense_interp = false;
    if (display_dense_interp) {
        float scaleFactor = 256.0f / max(originalWidth, originalHeight);
        int denseInterpWidth = scaleFactor * originalWidth;
        int denseInterpHeight = scaleFactor * originalHeight;

        SparseInterp<double> interp(denseInterpWidth, denseInterpHeight);
        
        interp.init(reconstruct.getPointCount(), 1.0);

        const int minInlierCount = 1;

        for (size_t i = 0; i < reconstruct.getPointCount(); i++) {
            double depth;
            size_t inlierC;

            const Eigen::Vector2d& pt = reconstruct.getDepthSample(i, depth, inlierC);

            if (depth > 0) {
                Eigen::Vector2d ptImg = (pt * max(denseInterpWidth, denseInterpHeight) / 2.0);

                ptImg.x() += denseInterpWidth / 2.0;
                ptImg.y() += denseInterpHeight / 2.0;

                interp.insertSample(i, ptImg.x() + 0.5, ptImg.y() + 0.5, depth);
            }
        }

        interp.solve();
    }
#endif


#if false
    bool display_dense_flow = false;
    if (display_dense_flow) {
        CImg<uint8_t> initDown = initImgGray;
        CImg<uint8_t> curDown;

        float scaleFactor = 1024.0f / max(originalWidth, originalHeight);
        int scaledWidth = scaleFactor * originalWidth;
        int scaledHeight = scaleFactor * originalHeight;

        // Resize with moving-average interpolation
        initDown.resize(scaledWidth, scaledHeight, -100, -100, 2);

        CVDenseOpticalFlow denseFlow;

        for (int imgI = 1; imgI < imageCount; imgI++) {
            printf("Computing dense flow for image #%d\n", imgI);

            curDown = CImg<uint8_t>::get_load(argv[1 + imgI]).RGBtoLab().channel(0);

            curDown.resize(scaledWidth, scaledHeight, -100, -100, 2);

            denseFlow.compute(initDown, curDown);

            CImg<float> flow(scaledWidth, scaledHeight, 2);

            cimg_forXY(flow, x, y) {
                denseFlow.getRelativeFlow(x, y, flow(x, y, 0), flow(x, y, 1));
            }

            (flow.get_shared_slice(0), flow.get_shared_slice(1)).display();
        }
    }
#endif

#if false
    bool display_all_depth_samples = false;
    if (display_all_depth_samples) {
        CImg<double> depthVisMed(workingWidth, workingHeight);

        vector<tuple<Eigen::Vector2d, double, double>> depthSamples;

        for (size_t cameraI = 0; cameraI < imageCount; cameraI++) {
            if (!reconstruct.isInlierCamera(cameraI)) {
                continue;
            }

            depthSamples.clear();

            reconstruct.getAllDepthSamples(cameraI, depthSamples);

            depthVisMed = 0.0f;

            for (auto& pointDepthSamples : depthSamples) {
                Eigen::Vector2d point = get<0>(pointDepthSamples);
                /*
                vector<tuple<double, double>>& depths = get<1>(pointDepthSamples);

                if (depths.size() < (imageCount - 1) / 2.0f) {
                    continue;
                }
                */

                point *= max(depthVisMed.width(), depthVisMed.height()) / 2.0;
                point.x() += depthVisMed.width() / 2.0;
                point.y() += depthVisMed.height() / 2.0;

                // TODO use nth element instead
                // std::sort(depths.begin(), depths.end());
                // d = get<0>(depths[depths.size() / 2]);

                double d = get<1>(pointDepthSamples);
                /*
                double highestConfidence = std::numeric_limits<double>::min();

                for (const auto& sample : depths) {
                    if (get<1>(sample) > highestConfidence) {
                        d = get<0>(sample);
                        highestConfidence = get<1>(sample);
                    }
                }
                */

                depthVisMed.draw_circle(point.x() + 0.5, point.y() + 0.5, 5, &d);
            }

            {
                float avgDepth = depthVisMed.sum() / depthVisMed.get_sign().abs().sum();
                depthVisMed -= (depthVisMed.get_sign().abs() - 1) * avgDepth;
            }

            bool renderEpipoles = false;

            if (renderEpipoles) {
                for (int imgI = 1; imgI < imageCount; imgI++) {
                    PolarFundamentalMatrix polarF;

                    bool rectificationPossible = reconstruct.getPolarFundamentalMatrix(
                            imgI - 1,
                            Eigen::Vector2d(depthVisMed.width() / 2.0, depthVisMed.height() / 2.0),
                            max(depthVisMed.width() / 2.0, depthVisMed.height() / 2.0),
                            polarF);

                    if (rectificationPossible) {
                        const Eigen::Vector2d& e = polarF.getEpipole(0);
                        double zero = 0;
                        depthVisMed.draw_circle(e.x(), e.y(), 20, &zero);
                    }
                }
            }

            depthVisMed.display();
        }
    }
#endif


#if false
    const bool display_rectification = false;
    if (display_rectification) {
        CImg<uint8_t> initDown = initImgGray;
        CImg<uint8_t> curDown;

        float scaleFactor = (512.0f) / max((float) originalWidth, (float) originalHeight);
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
#endif

    return 0;
}

