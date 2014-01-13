#include "Halide.h"

#include "adaptbp.h"

#include "cvutil/cvutil.h"

void computeDisparity(
        const CImg<float>& leftImg,
        const CImg<float>& rightImg,
        int minDisp,
        int maxDisp,
        CImg<float>& costImg,
        CImg<float>& disparityImg) {
    // TODO Use utin8_t instead of float wherever possible (everything before 'cost')
    Halide::ImageParam left(Halide::Float(32), 3, "left");
    Halide::ImageParam right(Halide::Float(32), 3, "right");

    Halide::Var x("x"), y("y"), c("c"), d("d");

    Halide::Param<float> omegaParam;
    Halide::Param<int> minDispParam;
    Halide::Param<int> maxDispParam;

    Halide::RDom rDisp(minDispParam, maxDispParam);
    Halide::RDom rC(0, 2);

    Halide::RDom r3x3(-1, 1, -1, 1);
    Halide::RDom r3x2(-1, 0, -1, 1);
    Halide::RDom r2x3(-1, 1, -1, 0);
    
    // C_SAD(x, y, c, d) ...
    Halide::Func absDiff("absDiff"), cSAD("cSAD");

    absDiff(x, y, c, d) = Halide::abs(left(x, y, c) - right(x + d, y, c));

    cSAD(x, y, c, d) += absDiff(x + r3x3.x, y + r3x3.y, c, d);

    // C_GRAD(x, y, c, d)...
    Halide::Func gradX1("gradX1"), gradX2("gradX2"), gradY1("gradY1"), gradY2("gradY2"), absGradX("absGradX"), absGradY("absGradY"), cGrad("cGrad");
    
    gradX1(x, y, c) = left(x + 1, y, c) - left(x, y, c);
    gradX2(x, y, c) = right(x + 1, y, c) - right(x, y, c);

    gradY1(x, y, c) = left(x, y + 1, c) - left(x, y, c);
    gradY2(x, y, c) = right(x, y + 1, c) - right(x, y, c);

    absGradX(x, y, c, d) = Halide::abs(gradX1(x, y, c) - gradX2(x + d, y, c));

    absGradY(x, y, c, d) = Halide::abs(gradY1(x, y, c) - gradY2(x + d, y, c));

    cGrad(x, y, c, d) +=
        absGradX(x + r3x2.x, y + r3x2.y, c, d) +
        absGradY(x + r3x2.x, y + r3x2.y, c, d);

    // C(x, y, d)...
    Halide::Func cost("cost");
    cost(x, y, d) += 
        (1.0f - omegaParam) * cSAD(x, y, rC, d) +
        omegaParam * cGrad(x, y, rC, d);

    // Argmin_d(x, y)...
    Halide::Func minCostDisparity("minCostDisparity");

    minCostDisparity(x, y) = 0;

    Halide::Expr bestDisparitySoFar =
        cost(x, y, clamp(minCostDisparity(x, y), minDispParam, maxDispParam));

    minCostDisparity(x, y) = select(cost(x, y, rDisp) < bestDisparitySoFar, rDisp, minCostDisparity(x, y));

    // Result(x, y)...
    // TODO Make this a tuple which includes the disparity
    Halide::Func result("result");
    result(x, y) = minCostDisparity(x, y);


    // Schedule...
    absDiff
        .compute_at(cSAD, x)
        .store_at(cSAD, y);

    cSAD
        .compute_at(cost, x)
        .store_at(cost, y);

    gradX1
        .compute_at(absGradX, x)
        .store_at(absGradX, y);
    gradX2
        .compute_at(absGradX, x)
        .store_at(absGradX, y);

    gradY1
        .compute_at(absGradY, x)
        .store_at(absGradY, y);
    gradY2
        .compute_at(absGradY, x)
        .store_at(absGradY, y);

    absGradX
        .compute_at(cGrad, y)
        .store_at(cGrad, y);

    absGradY
        .compute_at(cGrad, y)
        .store_at(cGrad, y);

    cGrad
        .compute_at(cost, x)
        .store_at(cost, y);

    cost
        .compute_at(minCostDisparity, y)
        .store_at(minCostDisparity, y)
        .bound(d, minDispParam, maxDispParam);

    minCostDisparity
        .compute_root()
        .bound(x, 0, left.width())
        .bound(y, 0, left.height());

    std::vector<Halide::Argument> args;
    args.push_back(left);
    args.push_back(right);
    args.push_back(minDispParam);
    args.push_back(maxDispParam);
    args.push_back(omegaParam);
    // minCostDisparity.compile_to_c("stereo_compiled.cpp", args, "stereo");
    minCostDisparity.compile_jit();
}

void computeAdaptBPStereo(
        const CImg<float>& left,
        const CImg<float>& right,
        int minDisp,
        int maxDisp,
        CImg<float>& dispLeft,
        CImg<float>& dispRight) {
    assert(left.is_sameXYZC(right));

    dispLeft = CImg<float>(left.width(), left.height());
    dispRight = CImg<float>(left.width(), left.height());

    dispLeft = std::numeric_limits<float>::max();
    dispRight = std::numeric_limits<float>::max();

    /**
     * Compute a segmentation via Slic superpixelization.
     *
     * Note that this differs from the original paper, which used
     * Mean-shift color segmentation (Comaniciu and Meer)
     */
    CImg<int> spLeft, spRight;

    int numSuperpixels = (left.width() * left.height()) / 256;

    printf("Computing superpixels for Left\n");
    // slicSuperpixels(left, numSuperpixels, 15, spLeft);

    printf("Computing superpixels for Right\n");
    // slicSuperpixels(right, numSuperpixels, 15, spRight);

    CImg<float> cSADLeft(maxDisp - minDisp, left.width());
    CImg<float> cGRADLeft(maxDisp - minDisp, left.width());

    CImg<float> cSADRight(maxDisp - minDisp, left.width());
    CImg<float> cGRADRight(maxDisp - minDisp, left.width());

    cSADLeft = std::numeric_limits<float>::max();
    cGRADLeft = std::numeric_limits<float>::max();

    CImg<float> costLeft(left.width(), left.height());

    computeDisparity(left, right, minDisp, maxDisp, costLeft, dispLeft);
}
