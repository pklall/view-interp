#include "Halide.h"

#include "adaptbp.h"

#include "cvutil/cvutil.h"

#include <tuple>

using namespace std;

void computeDisparity(
        const CImg<int16_t>& leftImg,
        const CImg<int16_t>& rightImg,
        int minDisp,
        int maxDisp,
        float omega,
        CImg<int16_t>& disparityImg) {
    assert(leftImg.is_sameXYZC(rightImg));
    
    Halide::Buffer leftBuf(
            Halide::Int(16),
            leftImg.width(), leftImg.height(),
            1, leftImg.spectrum(),
            (uint8_t*) leftImg.data(), string("leftBuf"));

    Halide::Buffer rightBuf(
            Halide::Int(16),
            rightImg.width(), rightImg.height(),
            1, rightImg.spectrum(),
            (uint8_t*) rightImg.data(), string("rightBuf"));

    // Inputs
    Halide::Image<int16_t> left(leftBuf.raw_buffer(), string("left"));
    Halide::Image<int16_t> right(rightBuf.raw_buffer(), string("right"));

    Halide::Param<float> omegaParam;
    Halide::Param<int> minDispParam;
    Halide::Param<int> maxDispParam;

    // TODO omega should be dynamically chosen
    omegaParam.set(omega);
    minDispParam.set(minDisp);
    maxDispParam.set(maxDisp);

    // Variables
    
    Halide::Var x("x"), y("y"), c("c"), d("d");
    
    Halide::RDom rDisp(minDispParam, maxDispParam);
    Halide::RDom rC(0, 2);

    int wndRad = 1;
    Halide::RDom r3x3(-wndRad, wndRad, -wndRad, wndRad);
    Halide::RDom r3x2(-wndRad, 0, -wndRad, wndRad);
    Halide::RDom r2x3(-wndRad, wndRad, -wndRad, 0);

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
    // TODO Robustify cSAD and cGrad with a max distance learned from optimization
    cost(x, y, d) += 
        (1.0f - omegaParam) * Halide::cast(Halide::Float(32), cSAD(x, y, rC, d)) +
        omegaParam * Halide::cast(Halide::Float(32), cGrad(x, y, rC, d));

    // Argmin_d(x, y)...
    Halide::Func minCostDisparity("minCostDisparity");

    minCostDisparity(x, y) = Halide::cast(Halide::Int(16), 0);

    Halide::Expr bestDisparitySoFar =
        cost(x, y, clamp(minCostDisparity(x, y), minDispParam, maxDispParam));

    minCostDisparity(x, y) = select(
            cost(x, y, rDisp) < bestDisparitySoFar,
            Halide::cast(Halide::Int(16), rDisp),
            minCostDisparity(x, y));
    
    // Argmin_d_reverse(x, y)...
    Halide::Func costRev("costRev");

    costRev(x, y, d) = cost(x + d, y, -d);

    Halide::Func minCostDisparityRev("minCostDisparityRev");

    minCostDisparityRev(x, y) = Halide::cast(Halide::Int(16), 0);

    Halide::Expr bestDisparitySoFarRev =
        costRev(x, y, clamp(minCostDisparityRev(x, y), -maxDispParam, -minDispParam));

    minCostDisparityRev(x, y) = select(
            costRev(x, y, -rDisp) < bestDisparitySoFarRev,
            Halide::cast(Halide::Int(16), -rDisp),
            minCostDisparityRev(x, y));

    // Holes(x, y)...
    Halide::Expr revX = minCostDisparity(x, y) + x;
    Halide::Expr revXC = Halide::clamp(revX, 0, left.width() - 1);

    Halide::Expr consistent = Halide::select(revXC == revX,
            Halide::abs(
                minCostDisparityRev(revXC, y) + minCostDisparity(x, y)
                ) < 2.0f,
            false);

    // Result(x, y)...
    Halide::Func result("result");

    // TODO Only compute over valid range of x values (ignore left & right border)
    Halide::Expr validX = Halide::clamp(
            x,
            Halide::max(0, -minDispParam),
            Halide::min(left.width() - 1, left.width() - maxDispParam));
    
    result(x, y) =
        Halide::select(consistent,
                Halide::cast(Halide::Int(16),
                    Halide::select(x == validX,
                        minCostDisparity(validX, y),
                        Halide::Int(16).max()
                        )
                    ),
                Halide::Int(16).max());

    // Schedule...
    absDiff
        .reorder(x, c, d, y)
        .compute_at(cSAD, x)
        .store_at(cSAD, d)
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
        .compute_at(result, y)
        .store_root();

    minCostDisparity
        .reorder(x, y)
        .compute_at(result, y)
        .store_root()
        .bound(x, 0, left.width())
        .bound(y, 0, left.height());

    minCostDisparityRev
        .reorder(x, y)
        .compute_at(result, y)
        .store_root()
        .bound(x, 0, left.width())
        .bound(y, 0, left.height());

    result.compute_inline()
        .reorder(x, y);

    vector<Halide::Argument> args;
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

    // Copy out the results from buffer to CImg
    {
        Halide::Buffer disparityBuf = r[0];
        Halide::Image<int16_t> disparityI(disparityBuf);

        disparityImg= CImg<int16_t>(leftImg.width(), leftImg.height());
        cimg_forXY(disparityImg, x, y) {
            disparityImg(x, y) = disparityI(x, y);
        }

    }
}

void fitPlanes(
        const CImg<int>& segmentation,
        const CImg<int16_t>& disp,
        vector<Plane>& planes) {
    assert(segmentation.is_sameXYZC(disp));

    CImg<float> dx(disp.width(), disp.height());
    dx = -100.0f;

    vector<tuple<float, float>> dSamples;

    for (int y = 0; y < disp.height(); y++) {
        int curSegment = -1;

        dSamples.clear();

        for (int x = 0; x < disp.width(); x++) {
            // If we've reached a new segment
            if (segmentation(x, y) != curSegment) {
                int n = dSamples.size() - 1;

                // We can only estimate planes with more than 1 sample in
                // the segment
                if (n > 1) {
                    int numReliablePairs = (n * (n + 1) / 2);

                    CImg<float> dxSamples(numReliablePairs);

                    int index = 0;
                    for (int i = 0; i < dSamples.size(); i++) {
                        for (int j = i + 1; j < dSamples.size(); j++) {
                            dxSamples(index) = 
                                (get<1>(dSamples[j]) - get<1>(dSamples[i])) /
                                (get<0>(dSamples[j]) - get<0>(dSamples[i]));
                            index++;
                        }
                    }

                    dxSamples.sort();

                    // FIXME The paper doesn't specify blur kernel size!
                    dxSamples.blur(2.0f);

                    int estimatedDx = dxSamples(dxSamples.size() / 2);

                    // Write our estimated x-slant to all pixels in the segment
                    for (int xp = x - 1; segmentation(xp, y) == curSegment; xp--) {
                        dx(xp, y) = estimatedDx;
                    }
                }

                curSegment = segmentation(x, y);

                dSamples.clear();
            } else if (disp(x, y) != std::numeric_limits<int16_t>::max()) {
                dSamples.push_back(make_tuple((float) x, (float) disp(x, y)));
            }
        }
    }

    disp.display();
    dx.display();
}

void computeAdaptBPStereo(
        const CImg<int16_t>& left,
        const CImg<int16_t>& right,
        int minDisp,
        int maxDisp,
        CImg<int16_t>& disp) {
    assert(left.is_sameXYZC(right));

    /**
     * Compute a segmentation via Slic superpixelization.
     *
     * Note that this differs from the original paper, which used
     * Mean-shift color segmentation (Comaniciu and Meer)
     */
    CImg<int> superpixels;

    int numSuperpixels = (left.width() * left.height()) / 256;

    printf("Computing superpixels for Left\n");
    slicSuperpixels(left, numSuperpixels, 15, superpixels);

    computeDisparity(left.get_RGBtoLab(), right.get_RGBtoLab(),
            minDisp, maxDisp, 0.5f, disp);

    vector<Plane> planes;
    fitPlanes(superpixels, disp, planes);
}
