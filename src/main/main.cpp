#include "common.h"

#include "util.h"

#include "pmstereo/pmstereo.h"

#include "cvstereo/cvstereo.h"

#include <algorithm>

// #define LBFGS_FLOAT 32
#include <lbfgs.h>

using namespace std;

using namespace cimg_library;

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

void optimizeLBFGS(
        int N,
        std::function<void(lbfgsfloatval_t*)> initGuess,
        std::function<lbfgsfloatval_t(const lbfgsfloatval_t*, lbfgsfloatval_t*)> eval,
        std::function<void(const lbfgsfloatval_t*, lbfgsfloatval_t)> result) {
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N);
    lbfgs_parameter_t param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return;
    }

    initGuess(x);

    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);
    // param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

    auto evaluate = [](
            void *instance,
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *g,
            const int n,
            const lbfgsfloatval_t step
            ) -> lbfgsfloatval_t {
        auto eval = (std::function<lbfgsfloatval_t(const lbfgsfloatval_t*, lbfgsfloatval_t*)>*) instance;
        return (*eval)(x, g);
    };

    auto progress = [](
            void *instance,
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls) -> int{
        // Do nothing
        return 0;
    };

    auto evalCopy = eval;
    
    int ret = lbfgs(N, x, &fx, evaluate, progress, &evalCopy, &param);
    printf("L-BFGS optimization terminated with status code = %d\n", ret);

    result(x, fx);

    lbfgs_free(x);
}

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
    
    // CImgList<float> foo(fst, lst);
    // foo.display();

    // Goal:
    //  * Known: a, b, offset, alpha
    //  * Unknown: f, g
    //  * Constraints:
    //      // FIXME Add scaling constant to SSD terms?
    //      sqr(f(x, y) + g(x, y) * alpha - a(x, y)) == 0
    //      sqr(f(x, y) + g(x + offset, y) * alpha - b(x, y)) == 0
    //      f(x, y) - f(x + 1, y    ) == 0
    //      f(x, y) - f(x - 1, y    ) == 0
    //      f(x, y) - f(x,     y + 1) == 0
    //      f(x, y) - f(x,     y - 1) == 0
    //      g(x, y) - g(x + 1, y    ) == 0
    //      g(x, y) - g(x - 1, y    ) == 0
    //      g(x, y) - g(x,     y + 1) == 0
    //      g(x, y) - g(x,     y - 1) == 0

    int N = a.size() + b.size();

    auto initGuess = [&](lbfgsfloatval_t* X) {
        CImg<double> f(X, a.width(), a.height(), a.depth(), a.spectrum(), true);
        CImg<double> g(X + f.size(), a.width(), a.height(), a.depth(), a.spectrum(), true);

        f = b;
        g = a;
    };


    // TODO Speed this up with OpenMP
    auto eval = [&](
            const lbfgsfloatval_t* X,
            lbfgsfloatval_t* G) -> lbfgsfloatval_t {
        // Wrap X and G with shared-memory images
        const CImg<double> f(X, a.width(), a.height(), a.depth(), a.spectrum(), true);
        const CImg<double> g(X + f.size(), a.width(), a.height(), a.depth(), a.spectrum(), true);
        CImg<double> dF(G, a.width(), a.height(), a.depth(), a.spectrum(), true);
        CImg<double> dG(G + dF.size(), a.width(), a.height(), a.depth(), a.spectrum(), true);

        const int neighborhood[4][2] {
            {0, 1}, {0, -1}, {-1, 0}, {1, 0}
        };

        double totalCost = 0.0f;
        dF = 0.0f;
        dG = 0.0f;

#pragma omp parallel for
        for (int c = 0; c < a.spectrum(); c++) {
            cimg_forXYZ(a, x, y, z) {
                if (!g.containsXYZC(x + offset, y, 0, c)) {
                    continue;
                }

                double diff;
                
                // Added constraint that results be close to 0.5
                diff = f(x, y, 0, c) - 0.5;
                totalCost += sqr(diff);
                dF(x, y, 0, c) += 2.0 * diff;

                diff = g(x, y, 0, c) - 0.5;
                totalCost += sqr(diff);
                dG(x, y, 0, c) += 2.0 * diff;

                double weight = 1000.0;
                diff = f(x, y, 0, c) * (1.0 - alpha) + g(x, y, 0, c) * alpha - a(x, y, 0, c);
                totalCost += weight * sqr(diff);
                dF(x, y, 0, c) += weight * 2.0 * diff * (1.0 - alpha);
                dG(x, y, 0, c) += weight * 2.0 * diff * alpha;

                diff = f(x, y, 0, c) * (1.0 - alpha) + g(x + offset, y, 0, c) * alpha - b(x, y, 0, c);
                totalCost += weight * sqr(diff);
                dF(x, y, 0, c) += weight * 2.0 * diff * (1.0 - alpha);
                dG(x + offset, y, 0, c) += weight * 2.0 * diff * alpha;

                for (int i = 0; i < 4; i++) {
                    int nx = x + neighborhood[i][0];
                    int ny = y + neighborhood[i][1];

                    if (nx >= 0 && nx < a.width() && ny >= 0 && ny < a.height()) {
                        diff = f(x, y, 0, c) - f(nx, ny, 0, c);
                        totalCost += sqr(diff);
                        dF(x, y, 0, c) += 2.0 * (diff) * 1.0;
                        dF(nx, ny, 0, c) += 2.0 * (diff) * -1.0;

                        diff = g(x, y, 0, c) - g(nx, ny, 0, c);
                        totalCost += sqr(diff);
                        dG(x, y, 0, c) += 2.0 * diff * 1.0;
                        dG(nx, ny, 0, c) += 2.0 * diff * -1.0;
                    }
                }
            }
        }

        printf("Cost = %f\n", totalCost);

        return totalCost;
    };


    auto result = [&](const lbfgsfloatval_t* X, lbfgsfloatval_t error) {
        CImg<double> f(X, a.width(), a.height(), a.depth(), a.spectrum(), false);
        CImg<double> g(X + f.size(), a.width(), a.height(), a.depth(), a.spectrum(), false);

        f.display();
        g.display();
    };

    optimizeLBFGS(N, initGuess, eval, result);
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


