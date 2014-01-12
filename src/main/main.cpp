#include "ceres/ceres.h"

#include "glog/logging.h"

#include "common.h"

#include "util.h"

#include "pmstereo/pmstereo.h"

#include "cvstereo/cvstereo.h"

#include <algorithm>

using namespace std;

using namespace cimg_library;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#define dbgOut std::cout

void runStereoMatte(CImg<float>& fst, CImg<float>& lst);

void runCVStereo(CImg<float>& fst, CImg<float>& lst);

void runPMStereo(CImg<float>& fst, CImg<float>& lst);

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <operation> left.png right.png\n", argv[0]);
        exit(1);
    }

    string op(argv[1]);
    CImg<float> fst(argv[2]);
    CImg<float> lst(argv[3]);

    std::transform(op.begin(), op.end(),op.begin(), ::tolower);

    if (op == "cvstereo") {
        printf("Running CVStereo\n");
        runCVStereo(fst, lst);
    } else if (op == "pmstereo") {
        runPMStereo(fst, lst);
    } else if (op == "stereomatte") {
        printf("Running stereomatte\n");
        runStereoMatte(fst, lst);
    }

    return 0;
}

struct StereoMattingCost {
    private:
        const CImg<float>& a, b;

    public:
        StereoMattingCost(const CImg<float>& _a, const CImg<float>& _b)
            : a(_a), b(_b)
        {
        }

        template <typename T> bool operator()(const T* const xRaw, T* residual) const {
            // residual[0] = T(10.0) - x[0];

            return true;
        }
};

void runStereoMatte(
        CImg<float>& fst,
        CImg<float>& lst) {
    fst.resize_halfXY();
    fst.resize_halfXY();
    lst.resize_halfXY();
    lst.resize_halfXY();

    CImg<double> a = fst;
    CImg<double> b = fst;

    // Remove alpha channel, if it exists.
    a.channels(0, 2);
    b.channels(0, 2);

    assert(a.is_sameXYZC(b));

    int offset = 20;
    double alpha = 0.5;

    a.draw_image(0, lst, alpha);
    b.draw_image(offset, lst.get_crop(0, 0, 0, 0, lst.width() - offset, lst.height(), lst.depth(), lst.spectrum()), alpha);

    assert(a.is_sameXYZC(b));

    CImgList<double>(a, b).display();

    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    CImg<double> x(a.width(), a.height(), a.depth(), a.spectrum());
    x.rand(0.0f, 255.0f);
    const CImg<double> initialX = x;

    // Build the problem.
    Problem problem;

    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).
    /*
    CostFunction* cost_function =
        new DynamicAutoDiffCostFunction<CostFunctor, 1, DYNAMIC>(new CostFunctor());
    problem.AddResidualBlock(cost_function, NULL, x.data());
    */

    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
}

void runCVStereo(
        CImg<float>& fst,
        CImg<float>& lst) {
    CVStereo stereo(fst, lst, true);

    CImg<float> rLeft, rRight;
    stereo.getRectified(rLeft, rRight);
    // CImgList<float>(rLeft, rRight).display();

    stereo.matchStereo();

    CImg<float> result;

    stereo.getStereo(result);

    result.display();
}

void runPMStereo(
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

