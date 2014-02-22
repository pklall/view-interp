#include <algorithm>
#include <fstream>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "util.h"

#include "adaptbp.h"

#include "dpstereo.h"

#include "cvutil/cvutil.h"

#include "common.h"

#include "time.h"

#include "reconstruct.h"

#include "rectify.h"

using namespace std;

using namespace cimg_library;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#define dbgOut std::cout

void runMultiview(
        const CImgList<uint8_t>& imgs);

void naiveStereoReconstruct(
        const CImg<float>& original,
        const CImg<float>& disparity,
        CImg<float>& result,
        float scale);

void runInterpolation(
        const CImg<float>& fst,
        const CImg<float>& lst);

void runBPStereo(CImg<int16_t>& fst, CImg<int16_t>& lst);

void runCVStereo(CImg<float>& fst, CImg<float>& lst);

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <operation> left.png right.png\n", argv[0]);
        exit(1);
    }

    string op(argv[1]);

    std::transform(op.begin(), op.end(),op.begin(), ::tolower);

    if (op == "cvstereo") {
        CImg<float> fst(argv[2]);
        CImg<float> lst(argv[3]);

        printf("Running CVStereo\n");
        runCVStereo(fst, lst);

    } else if (op == "bpstereo") {
        CImg<int16_t> fst(argv[2]);
        CImg<int16_t> lst(argv[3]);

        printf("Running bpstereo\n");
        runBPStereo(fst, lst);
    } else if (op == "interp") {
        CImg<float> fst(argv[2]);
        CImg<float> lst(argv[3]);

        printf("Running interpolation\n");
        runInterpolation(fst, lst);
    } else if (op == "multiview") {
        CImgList<uint8_t> imgs(argc - 2);

        for (int i = 2; i < argc; i++) {
            imgs(i - 2).load(argv[i]);
        }

        runMultiview(imgs);
    }

    return 0;
}

void runMultiview(
        const CImgList<uint8_t>& imgs) {
    ChainFeatureMatcher features;
    
    for (int i = 0; i < min((int) imgs.size(), 2); i++) {
        printf("Processing image %d\n", i);

        CImg<float> gray = imgs(i);

        if (gray.spectrum() > 1) {
            gray.RGBtoLab();
            gray.channel(0);
        }

        CImg<uint8_t> gray8 = gray;

        features.processNext(gray8);
    }

    features.visualizeFeatureMatches([imgs](int i) -> const CImg<uint8_t>& { return imgs(i); });

    Rectification cr(&features, Eigen::Vector2i(imgs(0).width(), imgs(0).height()));

    cr.solve(20);

    cr.print(cout);

    CImg<uint8_t> warpedL(imgs(0).width(), imgs(0).height(), 1, imgs(0).spectrum());
    CImg<uint8_t> warpedR(imgs(0).width(), imgs(0).height(), 1, imgs(0).spectrum());

    cr.warp(imgs(0), imgs(1), warpedL, warpedR);

    (warpedL, warpedR).display();
}

void runInterpolation(
        const CImg<float>& fst,
        const CImg<float>& lst) {
    int maxDisp = 256;
    int minDisp = -maxDisp;

    StereoProblem sp(fst, lst, minDisp, maxDisp);

    // Use OpenCV's StereoSGBM algorithm

    printf("Computing stereo...\n");
    CVStereo stereo(sp.leftLab, sp.rightLab, true);

    stereo.matchStereo(minDisp, maxDisp, 1, 1.0f);

    stereo.getStereo(sp.disp);

    printf("Done\n");


    printf("Computing segmentation\n");

    Segmentation segmentation;

    segmentation.createSlicSuperpixels(
            sp.leftLab,
            sp.disp.width() * sp.disp.height() / (8 * 8),
            10);

    printf("Done\n");

    // Save a visualization of the segmentation
    CImg<float> segVis;

    segmentation.renderVisualization(segVis);

    segVis.save("results/segmentation.png");

    printf("Computing connectivity\n");

    Connectivity connectivity;

    segmentation.getConnectivity(connectivity);

    printf("Done\n");

    PlanarDepth pd = PlanarDepth(&sp, &segmentation);

    pd.fitPlanesMedian();

    PlanarDepthSmoothingProblem pdRefine(
            &pd, &sp, &segmentation, &connectivity);

    pdRefine.computeInlierStats();

    CImg<float> disp;

    // Visualize unary cost:
    pd.fitPlanesMedian();
    pdRefine.visualizeUnaryCost(disp);
    disp.display();

    for (int iter = 0; iter < 10; iter++) {
        int s;
        cin >> s;

        float smoothness = s * 0.1f;

        pd.fitPlanesMedian();

        printf("Recomputing with smoothness = %f\n", smoothness);

        pdRefine.setSmoothness((float) smoothness);

        for (int i = 0; i < 3; i++) {
            pdRefine.solve();
        }

        pd.getDisparity(disp);

        string fname = "results/iteration_" + to_string(iter) + ".png";

        // disp.save(fname.c_str());

        // disp.display();
        (segVis, sp.disp, disp).display();
    }

    CImg<float> reconstruction;
    for (int i = 0; i <= 20; i++) {
        printf("Rendering %d\n", i);

        pd.renderInterpolated((i - 10.0f) / 10.0f, reconstruction);

        string fname = "results/reconstruction_" + to_string(i) + ".png";
        reconstruction.save(fname.c_str());
    }
}

void runBPStereo(
        CImg<int16_t>& fst,
        CImg<int16_t>& lst) {
    CImg<float> disp;

    int minDisp = -128;
    int maxDisp = 128;

    auto stereo = AdaptBPStereo(fst, lst, minDisp, maxDisp);
    stereo.computeStereo();
    stereo.getDisparity(disp);

    cimg_forXY(disp, x, y) {
        if (disp(x, y) < minDisp || disp(x, y) > maxDisp) {
            disp(x, y) = std::numeric_limits<float>::max();
        }
    }

    for (int i = 0; i <= 20; i++) {
        CImg<float> reconstruction;

        naiveStereoReconstruct(fst, disp, reconstruction, (i - 5.0f) / 10.0f);

        string fname = "results/reconstruction_" + to_string(i) + ".png";
        reconstruction.save(fname.c_str());
    }
}

void naiveStereoReconstruct(
        const CImg<float>& original,
        const CImg<float>& disparity,
        CImg<float>& result,
        float scale = 1.0f) {
    CImg<int> sorted;

    disparity.get_sort(sorted, false);

    result = original;

    result = 0.0f;
    
    cimg_forXY(sorted, x, y) {
        int sx = (sorted(x, y) % disparity.width());
        int sy = (sorted(x, y) / disparity.width());

        if (disparity(sx, sy) != std::numeric_limits<float>::max()) {
            float dx = sx - scale * disparity(sx, sy);
            float dy = sy;

            if (original.containsXYZC(sx, sy) && result.containsXYZC(dx, dy)) {
                cimg_forZC(original, z, c) {
                    result(dx, dy, z, c) = original(sx, sy, z, c);
                }
            }
        }
    }
}

void runCVStereo(
        CImg<float>& fst,
        CImg<float>& lst) {
    CImg<float> dispLeft, dispRight;

    int maxDisp = 256;

    printf("Computing stereo...\n");
    {
        CVStereo stereo(fst, lst, true);
        stereo.matchStereo(-maxDisp, maxDisp, 3, 1.0f);
        stereo.getStereo(dispLeft);
        dispLeft.display();
    }
    printf("Done\n");

    printf("Computing stereo...\n");
    {
        CVStereo stereo(lst, fst, true);
        stereo.matchStereo(-maxDisp, maxDisp, 3, 1.0f);
        stereo.getStereo(dispRight);
    }
    printf("Done\n");

    // Use infinity to signify lack of data
    cimg_forXY(dispLeft, x, y) {
        if (dispLeft(x, y) < -maxDisp) {
            dispLeft(x, y) = std::numeric_limits<float>::max();
        }
    }

    cimg_forXY(dispRight, x, y) {
        if (dispRight(x, y) < -maxDisp) {
            dispRight(x, y) = std::numeric_limits<float>::max();
        }
    }

    for (int i = 0; i <= 10; i++) {
        CImg<float> reconstruction;

        naiveStereoReconstruct(fst, dispLeft, reconstruction, i / 10.0f);

        string fname = "results/reconstruction_" + to_string(i) + ".png";
        reconstruction.save(fname.c_str());
    }

    for (int i = 11; i <= 20; i++) {
        CImg<float> reconstruction;

        naiveStereoReconstruct(lst, dispRight, reconstruction, (i - 10) / 10.0f);

        string fname = "results/reconstruction_" + to_string(i) + ".png";
        reconstruction.save(fname.c_str());
    }
}

