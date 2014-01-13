#include "Halide.h"

#include "adaptbp.h"

#include "cvutil/cvutil.h"

void computeDisparity(
        const CImg<uint8_t>& leftImg,
        const CImg<uint8_t>& rightImg,
        int minDisp,
        int maxDisp,
        CImg<float>& costImg,
        CImg<float>& disparityImg) {
    // TODO Use utin8_t instead of float wherever possible (everything before 'cost')
    // Halide::ImageParam left(Halide::Float(32), 3, "left");
    // Halide::ImageParam right(Halide::Float(32), 3, "right");
    
    Halide::Buffer leftBuf(
            Halide::Float(32),
            leftImg.width(), leftImg.height(),
            1, leftImg.spectrum(),
            (uint8_t*) leftImg.data(), std::string("leftBuf"));

    Halide::Buffer rightBuf(
            Halide::Float(32),
            rightImg.width(), rightImg.height(),
            1, rightImg.spectrum(),
            (uint8_t*) rightImg.data(), std::string("rightBuf"));

    // Inputs

    Halide::Image<float> left(leftBuf.raw_buffer(), std::string("left"));
    Halide::Image<float> right(rightBuf.raw_buffer(), std::string("right"));

    Halide::Param<float> omegaParam;
    Halide::Param<int> minDispParam;
    Halide::Param<int> maxDispParam;

    omegaParam.set(0.5f);
    minDispParam.set(minDisp);
    maxDispParam.set(maxDisp);

    // Variables
    
    Halide::Var x("x"), y("y"), c("c"), d("d");
    
    Halide::RDom rDisp(minDispParam, maxDispParam);
    Halide::RDom rC(0, 2);

    Halide::RDom r3x3(-1, 1, -1, 1);
    Halide::RDom r3x2(-1, 0, -1, 1);
    Halide::RDom r2x3(-1, 1, -1, 0);

    // Helper expressions to clamp to image bounds
    
    Halide::Expr cx = clamp(x,
            max(-minDispParam, 1),
            min(left.width() - maxDispParam, left.width() - 2));

    Halide::Expr cy = clamp(y, 1, left.height() - 2);

    Halide::Func leftC("leftC");
    Halide::Func rightC("rightC");
    leftC(x, y, c) = left(cx, cy, 0, c);
    rightC(x, y, c) = right(cx, cy, 0, c);
    
    // C_SAD(x, y, c, d) ...
    Halide::Func absDiff("absDiff"), cSAD("cSAD");

    absDiff(x, y, c, d) = Halide::abs(leftC(x, y, c) - rightC(x + d, y, c));

    cSAD(x, y, c, d) += absDiff(x + r3x3.x, y + r3x3.y, c, d);

    // C_GRAD(x, y, c, d)...
    Halide::Func gradX1("gradX1"), gradX2("gradX2");
    Halide::Func gradY1("gradY1"), gradY2("gradY2");
    Halide::Func absGradX("absGradX"), absGradY("absGradY");
    Halide::Func cGrad("cGrad");
    
    gradX1(x, y, c) = leftC(x + 1, y, c) - leftC(x, y, c);
    gradX2(x, y, c) = rightC(x + 1, y, c) - rightC(x, y, c);

    gradY1(x, y, c) = leftC(x, y + 1, c) - leftC(x, y, c);
    gradY2(x, y, c) = rightC(x, y + 1, c) - rightC(x, y, c);

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

    minCostDisparity(x, y) = select(
            cost(x, y, rDisp) < bestDisparitySoFar,
            rDisp,
            minCostDisparity(x, y));

    // Result(x, y)...
    // TODO Make this a tuple which includes the disparity
    Halide::Func result("result");

    result(x, y) = minCostDisparity(x, y);


    // Schedule...
    absDiff
        .reorder(x, c, d, y)
        .compute_at(cSAD, x)
        .store_root()
        .bound(c, 0, 2);

    cSAD
        .reorder(x, c, d, y)
        .compute_at(cost, x)
        .store_root()
        .bound(c, 0, 2);

    gradX1
        .reorder(x, c, y)
        .compute_at(absGradX, x)
        .store_root()
        .bound(c, 0, 2);

    gradX2
        .reorder(x, c, y)
        .compute_at(absGradX, x)
        .store_root()
        .bound(c, 0, 2);

    gradY1
        .reorder(x, c, y)
        .compute_at(absGradY, x)
        .store_root()
        .bound(c, 0, 2);

    gradY2
        .reorder(x, c, y)
        .compute_at(absGradY, x)
        .store_root()
        .bound(c, 0, 2);

    absGradX
        .reorder(x, c, y)
        .compute_at(cGrad, x)
        .store_root()
        .bound(c, 0, 2);

    absGradY
        .reorder(x, c, y)
        .compute_at(cGrad, x)
        .store_root()
        .bound(c, 0, 2);

    cGrad
        .reorder(x, c, d, y)
        .compute_at(cost, x)
        .store_root();

    cost
        .reorder(x, y)
        .compute_at(minCostDisparity, y)
        .store_root();

    minCostDisparity
        .reorder(x, y)
        .compute_at(result, y)
        .store_root()
        .bound(x, 0, left.width())
        .bound(y, 0, left.height());

    result.compute_root()
        .reorder(x, y);

    std::vector<Halide::Argument> args;
    args.push_back(left);
    args.push_back(right);
    args.push_back(minDispParam);
    args.push_back(maxDispParam);
    args.push_back(omegaParam);

    printf("Compiling...\n");
    // result.compile_to_c("result.cpp", args);
    result.compile_jit();
    printf("Done\n");

    printf("Running...\n");
    Halide::Realization r = result.realize(leftImg.width(), leftImg.height());
    printf("Done...\n");

    // TODO Run it!
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
