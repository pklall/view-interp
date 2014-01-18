#include "adaptbp.h"

#include "cvutil/cvutil.h"

#include <tuple>
#include <vector>
#include <list>
#include <map>

using namespace std;

namespace ogm = opengm;

AdaptBPStereo::AdaptBPStereo(
        const CImg<int16_t>& _left,
        const CImg<int16_t>& _right,
        int _minDisp,
        int _maxDisp) :
    left(_left),
    right(_right),
    minDisp(_minDisp),
    maxDisp(_maxDisp),
    omega(0.5f),
    smoothFactor(10.0f),
    numSuperpixels(256) {
}

void AdaptBPStereo::computeGreedyDisp() {
    assert(left.is_sameXYZC(right));
    
    Halide::Buffer leftBuf(
            Halide::Int(16),
            left.width(), left.height(),
            1, left.spectrum(),
            (uint8_t*) left.data(), string("leftBuf"));

    Halide::Buffer rightBuf(
            Halide::Int(16),
            right.width(), right.height(),
            1, right.spectrum(),
            (uint8_t*) right.data(), string("rightBuf"));

    // Inputs
    Halide::Image<int16_t> leftImg(leftBuf.raw_buffer(), string("leftImg"));
    Halide::Image<int16_t> rightImg(rightBuf.raw_buffer(), string("rightImg"));

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
    // TODO what's the overhead of this?  check compiler output...
    leftC(x, y, c) = leftImg(cx, cy, 0, c);
    rightC(x, y, c) = rightImg(cx, cy, 0, c);
    
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
        .reorder_storage(x, c, d, y)
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
        .reorder_storage(x, c, d, y)
        .bound(c, 0, 2)
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
    args.push_back(leftImg);
    args.push_back(rightImg);
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
        Halide::Image<int16_t> disparityImg(disparityBuf);

        disp = CImg<int16_t>(left.width(), left.height());
        cimg_forXY(disp, x, y) {
            disp(x, y) = disparityImg(x, y);
        }

    }
}

/**
 * Estimates slant (dD/dt) from a set of samples of (D, t)
 * samples for which all non-t dimensions are constant.
 *
 * For example, if dSamples contains a set of (D, x) samples
 * from the same horizontal scan-line (fixed y-coordinate),
 * it will estimate dD/dx.
 *
 * Returns true upon success, false if not enough samples were provided.
 */
inline bool AdaptBPStereo::estimateSlant(
        const map<uint16_t, vector<tuple<uint16_t, int16_t>>>& dSamples,
        float& result) {
    int totalSamplePairs = 0;
    for (const auto& samples : dSamples) {
        int n = samples.second.size() - 1;
        totalSamplePairs += n * (n + 1) / 2;
    }

    if (totalSamplePairs < 1) {
        return false;
    }

    // Store all possible samples of dt, each consisting of a finite difference
    // between elements in the same scanline, as given by dSamples.
    CImg<float> dtSamples(totalSamplePairs);

    // Index into dtSamples at which to insert new samples
    int dtSamplesI = 0;

    for (const auto& samplesPair: dSamples) {
        const vector<tuple<uint16_t, int16_t>>& samples = samplesPair.second;

        for (int i = 0; i < samples.size(); i++) {
            for (int j = i + 1; j < samples.size(); j++) {
                assert(dtSamplesI < dtSamples.width());
                dtSamples(dtSamplesI) = 
                    ((float) get<1>(samples[j]) - get<1>(samples[i])) /
                    ((float) get<0>(samples[j]) - get<0>(samples[i]));
                dtSamplesI++;
            }
        }
    }

    dtSamples.sort();

    // TODO The paper doesn't specify blur kernel size!
    // dtSamples.blur(min(1.0f, dtSamples.size() / 6.0f));

    result = dtSamples(dtSamples.size() / 2);

    return true;
}


void AdaptBPStereo::fitPlanes() {
    // Create a plane for each superpixel
    planes.clear();

    // A map from y-index to (x, disparity) tuples to store
    // valid disparities for each scan-line in a superpixel.
    map<uint16_t, vector<tuple<uint16_t, int16_t>>> xDSamples;
    
    // A map from x-index to (y, disparity) tuples to store
    // valid disparities for each vertical-line in a superpixel.
    map<uint16_t, vector<tuple<uint16_t, int16_t>>> yDSamples;

    for (int superpixelI = 0; superpixelI < superpixels.size(); superpixelI++) {
        const auto& pixels = superpixels[superpixelI];

        xDSamples.clear();
        yDSamples.clear();

        int numValidD = 0;

        // Iterate over all pixels within the superpixel
        for (const auto& p : pixels) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            // If this pixel has a valid disparity, add it
            if (disp(x, y) != std::numeric_limits<int16_t>::max()) {
                xDSamples[y].push_back(make_tuple(x, disp(x, y)));
                yDSamples[x].push_back(make_tuple(y, disp(x, y)));

                numValidD++;
            }
        }
        
        float cx, cy;

        if (!estimateSlant(xDSamples, cx)) {
            continue;
        }

        if (!estimateSlant(yDSamples, cy)) {
            continue;
        }

        CImg<float> cSamples(numValidD);

        int cSamplesI = 0;

        // Iterate again, collecting samples with which to estimate
        // the 'c' value for the plane
        for (const auto& p : pixels) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            if (disp(x, y) != std::numeric_limits<int16_t>::max()) {
                float c = disp(x, y) - (cx * x + cy * y);

                cSamples(cSamplesI) = c;
                cSamplesI++;
            }
        }

        cSamples.sort();
        
        // TODO Paper doesn't specify how much to blur
        // cSamples.blur(min(1.0f, cSamples.width() / 6.0f));

        float c = cSamples(cSamples.width() / 2);

        planes.push_back(Plane(cx, cy, c));
    }
}

void AdaptBPStereo::getDisparity(
        CImg<float>& disp) {
    disp = CImg<float>(left.width(), left.height());

    for (int superpixelI = 0; superpixelI < superpixels.size(); superpixelI++) {
        const auto& pixels = superpixels[superpixelI];

        // Iterate over all pixels within the superpixel
        for (const auto& p : pixels) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            disp(x, y) = planes[superpixelPlaneMap[superpixelI]].dispAt(x, y);
        }
    }
}

void AdaptBPStereo::computeSegmentPlaneCost() {
    int numPlanes = planes.size();
    int numSeg = superpixels.size();

    segmentPlaneCost = CImg<float>(numSeg, numPlanes);

    segmentPlaneCost = 0.0f;

    // '1' specifies forward finite differences
    CImgList<int16_t> leftGrad = left.get_gradient(0, 1);
    CImgList<int16_t> rightGrad = right.get_gradient(0, 1);

    // TODO Sort valid planes by the number of reliable samples used to compute them

    // TODO Optimize - Store bounding-box for superpixel, create early-out if
    //                 a plane transforms the bounding-box outside of valid range.
    for (int superpixelI = 0; superpixelI < numSeg; superpixelI++) {
        const auto& pixels = superpixels[superpixelI];

        for (int planeI = 0; planeI < planes.size(); planeI++) {
            const Plane& plane = planes[planeI];

            // Iterate over all pixels within the superpixel
            for (const auto& p : pixels) {
                uint16_t x = get<0>(p);
                uint16_t y = get<1>(p);

                float disp = plane.dispAt(x, y);

                if (disp > maxDisp || disp < minDisp) {
                    segmentPlaneCost(superpixelI, planeI) =
                        std::numeric_limits<float>::max();
                    break;
                }

                int rx = (int) (x + plane.dispAt(x, y) + 0.5f);
                int ry = y;

                if (rx < 0 || rx > right.width() - 2) {
                    segmentPlaneCost(superpixelI, planeI) =
                        std::numeric_limits<float>::max();
                    break;
                }

                float cost = 0;

                cimg_forZC(left, z, c) {
                    int16_t sad = abs(right(rx, ry, z, c) -
                            left(x, y, z, c));

                    int16_t grad = 0;

                    grad += abs(leftGrad(0)(x, y, z, c) -
                            rightGrad(0)(rx, ry, z, c));

                    grad += abs(leftGrad(1)(x, y, z, c) -
                            rightGrad(1)(rx, ry, z, c));

                    // TODO Robustify this by truncating against value
                    //      determined by the mean & sd of these for reliable
                    //      disparities found in the first step.

                    cost += (1.0f - omega) * sad + omega * grad;
                }

                segmentPlaneCost(superpixelI, planeI) += cost;
            }
        }
    }
}

void AdaptBPStereo::mergeSegmentsByPlane() {
    int numPlanes = planes.size();
    int numSeg = superpixels.size();

    // Map from each plane to the set of segments for which it is optimal
    map<int, vector<int>> planeSegments;

    for (int segmentI = 0; segmentI < numSeg; segmentI++) {
        int optimalPlaneI = 0;
        float optimalPlaneCost = std::numeric_limits<float>::max();

        for (int planeI = 0; planeI < numPlanes; planeI++) {
            float cost = segmentPlaneCost(segmentI, planeI);
            
            if (cost < optimalPlaneCost) {
                optimalPlaneCost = cost;
                optimalPlaneI = planeI;
            }
        }

        planeSegments[optimalPlaneI].push_back(segmentI);
    }
    
    // Create new superpixel vector by merging superpixels with the same optimal plane
    vector<vector<tuple<uint16_t, uint16_t>>> mergedSuperpixels(planeSegments.size());

    int mergedSegmentI = 0;

    for (const auto& ps : planeSegments) {
        const vector<int>& segments = ps.second;

        for (int segmentI : segments) {
            auto& segment = mergedSuperpixels[mergedSegmentI];

            segment.insert(segment.end(), superpixels[segmentI].begin(),
                    superpixels[segmentI].end());
        }

        mergedSegmentI++;
    }

    superpixels = mergedSuperpixels;

    // Recompute segmentation to match the updated superpixels
    for (int superpixelI = 0; superpixelI < superpixels.size(); superpixelI++) {
        const auto& pixels = superpixels[superpixelI];

        // Iterate over all pixels within the superpixel
        for (const auto& p : pixels) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            segmentation(x, y) = superpixelI;
        }
    }
}

void AdaptBPStereo::computeGreedySuperpixelPlaneMap() {
    int numPlanes = planes.size();
    int numSeg = superpixels.size();

    superpixelPlaneMap.clear();
    superpixelPlaneMap.reserve(numSeg);

    for (int segmentI = 0; segmentI < numSeg; segmentI++) {
        int optimalPlaneI = 0;
        float optimalPlaneCost = std::numeric_limits<float>::max();

        for (int planeI = 0; planeI < numPlanes; planeI++) {
            float cost = segmentPlaneCost(segmentI, planeI);
            
            if (cost < optimalPlaneCost) {
                optimalPlaneCost = cost;
                optimalPlaneI = planeI;
            }
        }

        superpixelPlaneMap[segmentI] = optimalPlaneI;
    }
}

void AdaptBPStereo::createMRF() {
    int numSegments = superpixels.size();
    int numPlanes = segmentPlaneCost.height();
    
    CImg<float> segTotCol(numSegments);

    segTotCol = 0.0f;

    // Lower-trianglular matrix containing the length of the border
    // between segments.
    CImg<int> borderLength(numSegments, numSegments);

    borderLength = 0;

    cimg_forXY(left, x, y) {
        cimg_forC(left, c) {
            segTotCol(segmentation(x, y)) += left(x, y, 0, c);
        }
    }

    for (int y = 1; y < left.height(); y++) {
        for (int x = 1; x < left.width(); x++) {
            int segA, segB;

            segA = segmentation(x - 1, y);
            segB = segmentation(x, y);

            borderLength(min(segA, segB), max(segA, segB))++;

            segA = segmentation(x, y - 1);
            segB = segmentation(x, y);

            borderLength(min(segA, segB), max(segA, segB))++;
        }
    }

    // 'numSegments' variables, each can take 'numPlanes' labels
    printf("Constructing MRF with %d variables and %d labels\n", numSegments, numPlanes);

    Space space(numSegments, numPlanes);
    mrf = GModel(space);

    for (int segI = 0; segI < numSegments; segI++) {
        // Pairwise terms...
        for (int segI2 = segI + 1; segI2 < numSegments; segI2++) {
            if (segI2 == segI) {
                continue;
            }

            int blength = borderLength(segI, segI2);

            if (blength > 0) {
                float meanCol1 = segTotCol(segI) / superpixels[segI].size();
                float meanCol2 = segTotCol(segI2) / superpixels[segI2].size();

                float colorSim = (1.0f - min(abs(meanCol1 - meanCol2), 255.0f) / 255.0f) * 0.5f
                    + 0.5f;

                // TODO double check this
                float pair = colorSim * blength * smoothFactor;

                ogm::PottsFunction<float> pairTerm(numPlanes, numPlanes, 0, pair);

                // vars[segI], vars[segI2], pair);
                GModel::FunctionIdentifier fid = mrf.addFunction(pairTerm);

                size_t vars[] = {(size_t) segI, (size_t) segI2};
                mrf.addFactor(fid, vars, vars + 2);
            }
        }

        // Data Term...

        size_t shape[] = {(size_t) numPlanes};

        ogm::ExplicitFunction<float> dataTerm(shape, shape + 1);

        for (int planeI = 0; planeI < numPlanes; planeI++) {
            dataTerm(planeI) = segmentPlaneCost(segI, planeI);
        }

        GModel::FunctionIdentifier fid = mrf.addFunction(dataTerm);

        size_t vars[] = {(size_t) segI};
        mrf.addFactor(fid, vars, vars + 1);
    }
} 

void AdaptBPStereo::solveMRF() {

    typedef ogm::MinSTCutBoost<size_t, float, ogm::PUSH_RELABEL> MinCutType;
    typedef ogm::GraphCut<GModel, ogm::Minimizer, MinCutType> MinGraphCut;
    typedef ogm::AlphaExpansion<GModel, MinGraphCut> MinAlphaExpansion;
    
    MinAlphaExpansion ae(mrf);
    
    ae.infer();

    superpixelPlaneMap = vector<size_t>(superpixels.size());
    ae.arg(superpixelPlaneMap);

    /*
    typedef ogm::BeliefPropagationUpdateRules<GModel, ogm::Minimizer> UpdateRules;
    typedef ogm::MessagePassing<GModel, ogm::Minimizer, UpdateRules,
            ogm::MaxDistance> BeliefPropagation;
    const size_t maxNumberOfIterations = 1;
    const double convergenceBound = 1e-7;
    const double damping = 0.0;
    BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
    BeliefPropagation bp(mrf, parameter);

    bp.setStartingPoint(superpixelPlaneMap.begin());

    BeliefPropagation::VerboseVisitorType visitor;
    bp.infer(visitor);

    superpixelPlaneMap = vector<size_t>(superpixels.size());
    bp.arg(superpixelPlaneMap);
    */
}

/**
 * Computes stereo correspondence based on 
 *
 * Segment-Based Stereo Matching Using Belief Propagation and a Self-Adapting
 * Dissimilarity Measure (by Klause, Sormann, and Karner)
 *
 * a.k.a "AdaptBP" in Middlebury rankings
 *
 */
void AdaptBPStereo::computeStereo() {
    assert(left.is_sameXYZC(right));

    /**
     * Compute a segmentation via Slic superpixelization.
     *
     * Note that this differs from the original paper, which used
     * Mean-shift color segmentation (Comaniciu and Meer)
     */
    printf("Computing superpixels for Left\n");
    slicSuperpixels(left.get_RGBtoLab(), numSuperpixels, 10, segmentation, superpixels);

    printf("Computing disparity\n");
    computeGreedyDisp();

    printf("Fitting planes...\n");
    fitPlanes();

    printf("Computing segment-plane cost...\n");
    computeSegmentPlaneCost();

    printf("Merging segments by plane\n");
    mergeSegmentsByPlane();

    fitPlanes();

    printf("Computing segment-plane cost\n");
    computeSegmentPlaneCost();

    computeGreedySuperpixelPlaneMap();

    createMRF();

    solveMRF();

    CImg<float> finalDisp;
    getDisparity(finalDisp);
    finalDisp.save(("results/smooth_factor_" + to_string(smoothFactor) + ".png").c_str());
}

