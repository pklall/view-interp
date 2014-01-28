#include "segment.h"

#include "cvutil/cvutil.h"

Segment::Segment() : totalLab{0.0f, 0.0f, 0.0f} {
    minX = std::numeric_limits<imageI_t>::max();
    minY = std::numeric_limits<imageI_t>::max();

    maxX = std::numeric_limits<imageI_t>::min();
    maxY = std::numeric_limits<imageI_t>::min();
}

void Connectivity::increment(
        segmentH_t a,
        segmentH_t b) {
    connectivity[a][b]++;
    connectivity[b][a]++;
}

int Connectivity::getConnectivity(
        segmentH_t a,
        segmentH_t b) const {
    auto foundA = connectivity.find(a);

    if (foundA == connectivity.end()) {
        return 0;
    } else {
        const auto& foundB = (*foundA).second.find(b);

        if (foundB == (*foundA).second.end()) {
            return 0;
        } else {
            return foundB->second;
        }
    }
}

void Segmentation::recomputeSegmentMap() {
    for (segmentH_t superpixelI = 0; superpixelI < superpixels.size(); superpixelI++) {
        const auto& pixels = superpixels[superpixelI].getPixels();

        for (const auto& p : pixels) {
            imageI_t x = get<0>(p);
            imageI_t y = get<1>(p);

            segmentMap(x, y) = superpixelI;
        }
    }
}

void Segmentation::createSlicSuperpixels(
        const CImg<float>& lab,
        int numSegments,
        int nc) {
    assert(lab.spectrum() == 3);

    slicSuperpixels(lab, numSegments, nc, segmentMap);

    superpixels = vector<Segment>(numSegments);

    // slicSuperpixels() results in degenerate segments without any
    // pixels.  To eliminate these, we need a mapping from
    // original handles to a contiguous set of handles to
    // valid segments.
    map<segmentH_t, segmentH_t> validSegmentHandles;

    segmentH_t curHandle = 0;

    float labTmp[3];

    cimg_forXY(segmentMap, x, y) {
        cimg_forC(lab, c) {
            labTmp[c] = lab(x, y, 0, c);
        }

        if (validSegmentHandles.count(segmentMap(x, y)) == 0) {
            validSegmentHandles[segmentMap(x, y)] = curHandle;

            curHandle++;
        }

        superpixels[validSegmentHandles[segmentMap(x, y)]].addPixel(x, y, labTmp);
    }

    superpixels.resize(curHandle);

    recomputeSegmentMap();

    for (Segment& s : superpixels) {
        s.compress();
    }

    for (const Segment& s : superpixels) {
        assert(s.size() > 0);
    }
}

void Segmentation::renderVisualization(
        CImg<float>& result) const {
    result.resize(segmentMap.width(), segmentMap.height(), 1, 3, -1);

    for (const Segment& sp : superpixels) {
        float lab[3];

        sp.avgLab(lab);

        cimg_forC(result, c) {
            for (const auto& coord : sp.getPixels()) {
                imageI_t x = get<0>(coord);
                imageI_t y = get<1>(coord);

                result(x, y, 0, c) = lab[c];
            }
        }
    }

    result.LabtoRGB();
}

void Segmentation::getConnectivity(
        Connectivity& c) const {
    c = Connectivity();

    for (int y = 1; y < segmentMap.height(); y++) {
        for (int x = 1; x < segmentMap.width(); x++) {
            segmentH_t segCur = segmentMap(x, y);
            segmentH_t segLeft = segmentMap(x - 1, y);
            segmentH_t segTop = segmentMap(x, y - 1);

            c.increment(segCur, segLeft);

            c.increment(segCur, segTop);
        }
    }
}

StereoProblem::StereoProblem(
        CImg<uint16_t> _left, 
        CImg<uint16_t> _right, 
        int _minDisp,
        int _maxDisp,
        CImg<float> _disp) :
    left(_left), right(_right),
    minDisp(_minDisp), maxDisp(_maxDisp),
    disp(_disp) {
}

bool PlanarDepth::tabulateSlantSamples(
        const map<imageI_t, vector<tuple<imageI_t, float>>>& dSamples,
        CImg<float>& dtSamples) const {
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
    dtSamples.resize(totalSamplePairs, 1, 1, 1, -1);

    // Index into dtSamples at which to insert new samples
    int dtSamplesI = 0;

    for (const auto& samplesPair: dSamples) {
        const vector<tuple<imageI_t, float>>& samples = samplesPair.second;

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

    return true;
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
inline bool PlanarDepth::estimateSlant(
        const map<imageI_t, vector<tuple<imageI_t, float>>>& dSamples,
        float& result) const {
    CImg<float> dtSamples;

    if (!tabulateSlantSamples(dSamples, dtSamples)) {
        return false;
    }

    // TODO The paper doesn't specify blur kernel size!
    // dtSamples.blur(min(1.0f, dtSamples.size() / 16.0f));

    result = dtSamples(dtSamples.size() / 2);

    return true;
}

void PlanarDepth::fitPlanesMedian() {
    planes.clear();

    // Allocate an invalid plane at 0
    planes.push_back(Plane());

    // A map from y-index to (x, disparity) tuples to store
    // valid disparities for each scan-line in a superpixel.
    map<imageI_t, vector<tuple<imageI_t, float>>> xDSamples;
    
    // A map from x-index to (y, disparity) tuples to store
    // valid disparities for each vertical-line in a superpixel.
    map<imageI_t, vector<tuple<imageI_t, float>>> yDSamples;

    for (int superpixelI = 0; superpixelI < segmentation->size(); superpixelI++) {
        // By default, all planes are mapped to the invalid plane at 0
        segmentPlaneMap[superpixelI] = 0;

        const auto& superpixel = (*segmentation)[superpixelI];

        xDSamples.clear();
        yDSamples.clear();

        int numValidD = 0;

        // Iterate over all pixels within the superpixel
        for (const auto& p : superpixel.getPixels()) {
            imageI_t x = get<0>(p);
            imageI_t y = get<1>(p);

            // If this pixel has a valid disparity, add it
            if (stereo->isValidDisp(x, y)) {
                xDSamples[y].push_back(make_tuple(x, stereo->disp(x, y)));
                yDSamples[x].push_back(make_tuple(y, stereo->disp(x, y)));

                numValidD++;
            }
        }

        printf("numValidD = %d\n", numValidD);
        
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
        for (const auto& p : superpixel.getPixels()) {
            imageI_t x = get<0>(p);
            imageI_t y = get<1>(p);

            if (stereo->isValidDisp(x, y)) {
                float c = stereo->disp(x, y) - (cx * x + cy * y);

                cSamples(cSamplesI) = c;
                cSamplesI++;
            }
        }

        cSamples.sort();
        
        // TODO Paper doesn't specify how much to blur
        // cSamples.blur(min(1.0f, cSamples.width() / 16.0f));

        float c = cSamples(cSamples.width() / 2);

        planes.push_back(Plane(cx, cy, c));
        segmentPlaneMap[superpixelI] = planes.size() - 1;
    }

    planes.shrink_to_fit();
}

float PlanarDepth::getPlaneCostL1(
        segmentH_t segment,
        const Plane& plane) const {
    if (!plane.isValid()) {
        return std::numeric_limits<float>::max();
    }

    const Segment& superpixel = (*segmentation)[segment];

    float l1Dist = 0;

    for (const auto& p : superpixel.getPixels()) {
        imageI_t x = get<0>(p);
        imageI_t y = get<1>(p);

        if (stereo->isValidDisp(x, y)) {
            float planeDisp = plane.dispAt((float) x, (float) y);

            float dispSample = stereo->disp(x, y);

            l1Dist += fabs(planeDisp - dispSample);
        }
    }

    return l1Dist;
}

PlanarDepth::PlanarDepth(
        const StereoProblem* _stereo,
        const Segmentation* _segmentation)
    : stereo(_stereo), segmentation(_segmentation) {
    planes = vector<Plane>(1);
    segmentPlaneMap = vector<planeH_t>(segmentation->size());
}

void PlanarDepth::getDisparity(
        CImg<float>& disp) const {
    disp = CImg<float>(stereo->left.width(), stereo->right.height());

    disp = 0.0f;

    for (segmentH_t superpixelI = 0; superpixelI < segmentation->size(); superpixelI++) {
        const auto& superpixel = (*segmentation)[superpixelI];

        const Plane& plane = getPlane(superpixelI);

        if (plane.isValid()) {
            for (const auto& p : superpixel.getPixels()) {
                imageI_t x = get<0>(p);
                imageI_t y = get<1>(p);

                disp(x, y) = plane.dispAt(x, y);
            }
        }
    }
}

void PlanarDepth::renderInterpolated(
        float t,
        CImg<float>& result) {
    vector<segmentH_t> segmentIndices(segmentation->size());

    for (segmentH_t i = 0; i < segmentation->size(); i++) {
        segmentIndices[i] = i;
    }

    // Sort segments (by index) according to depth at center
    // for back-to-front rendering (Painter's Algo.)
    std::sort(segmentIndices.begin(), segmentIndices.end(),
            [&](segmentH_t a, segmentH_t b) {
                int aX, aY, bX, bY;
                (*segmentation)[a].getCenter(aX, aY);
                (*segmentation)[b].getCenter(bX, bY);
                return getPlane(a).dispAt(aX, aY) < getPlane(b).dispAt(bX, bY);
            });

    // Add an alpha channel to the result
    result.resize(stereo->left.width(), stereo->left.height(), 1,
            stereo->left.spectrum(), -1);

    result = 0.0f;
    
    for (segmentH_t segI : segmentIndices) {
        const Segment& superpixel = (*segmentation)[segI];

        const Plane& plane = getPlane(segI);

        if (plane.isValid()) {
            int minX, minY, maxX, maxY;

            superpixel.getBounds(minX, minY, maxX, maxY);

            cimg_forC(stereo->left, c) {
                for (int y = minY; y <= maxY; y++) {
                    int minXD = (int) (minX + plane.dispAt(minX, y) * t);
                    int maxXD = (int) (maxX + plane.dispAt(maxX, y) * t + 0.5f);

                    minXD = max(0, minXD);
                    maxXD = min(result.width() - 1, maxXD);
                    
                    for (int dx = minXD; dx <= maxXD; dx++) {
                        float sx = (dx - t * (plane.c + plane.cy * y)) /
                            (1.0f + plane.cx * t);

                        sx = fmin(stereo->left.width() - 1, sx);
                        sx = fmax(0.0f, sx);
                        
                        int sxiL = (int) sx;
                        int sxiR = (int) (sx + 0.5f);

                        float sxiLweight = 1.0f - (sx - ((float) sxiL));
                        float sxiRweight = 1.0f - (((float) sxiR) - sx);

                        sxiLweight *= (*segmentation)(sxiL, y) == segI;
                        sxiRweight *= (*segmentation)(sxiR, y) == segI;

                        float totWeight = sxiLweight + sxiRweight;

                        if (totWeight > 0) {
                            sxiLweight /= totWeight;
                            sxiRweight /= totWeight;

                            result(dx, y, 0, c) = 
                                stereo->left(sxiL, y, 0, c) * sxiLweight +
                                stereo->left(sxiR, y, 0, c) * sxiRweight;
                        }
                    }
                }
            }
        }
    }
}

SegmentLabelProblem::SegmentLabelProblem(
        const Segmentation* _segmentation,
        size_t _numLabelsTotal,
        size_t _numLabelsPerSeg) :
    segmentation(_segmentation),
    numLabelsTotal(_numLabelsTotal),
    numLabelsPerSeg(_numLabelsPerSeg) {

    Space space(segmentation->size(), numLabelsPerSeg);

    model = GModel(space);
}

void SegmentLabelProblem::addUnaryFactor(
        segmentH_t segment,
        const map<planeH_t, float>& labelWeights) {
    assert(labelWeights.size() == numLabelsPerSeg);

    size_t shape[] = {(size_t) numLabelsPerSeg};

    ExplicitFunction dataTerm(begin(shape), end(shape));

    size_t index = 0;
    for (const auto& element : labelWeights) {
        planeH_t pH = element.first;
        float cost = element.second;
        
        planeIndexMap[make_tuple(segment, pH)] = index;
        indexPlaneMap[make_tuple(segment, index)] = pH;

        size_t coordinate[] = {index};

        dataTerm(coordinate) = cost;

        index++;
    }

    GModel::FunctionIdentifier fid = model.addFunction(dataTerm);

    size_t vars[] = {(size_t) numLabelsPerSeg};

    model.addFactor(fid, begin(vars), end(vars));
}

void SegmentLabelProblem::addBinaryFactor(
        segmentH_t segment1,
        segmentH_t segment2,
        function<float(planeH_t, planeH_t)> func) {
    // Wrap func to convert labels stored by the model into plane handles
    auto wrapper = [=](size_t val1, size_t val2) -> float {
        auto val1F = indexPlaneMap.find(make_tuple(segment1, val1));

        auto val2F = indexPlaneMap.find(make_tuple(segment2, val2));

        float result = func(val1F->second, val2F->second);

        return result;
    };

    CustomFunction pairTerm(numLabelsPerSeg, numLabelsPerSeg, wrapper);

    GModel::FunctionIdentifier fid = model.addFunction(pairTerm);

    size_t vars[] = {(size_t) segment1, (size_t) segment2};

    model.addFactor(fid, begin(vars), end(vars));
}

void SegmentLabelProblem::solveMAP(
        vector<planeH_t>& labels) {
    typedef opengm::MQPBO<GModel, opengm::Minimizer> MQPBO;

    MQPBO::Parameter params;
    params.useKovtunsMethod_ = false;

    MQPBO solver(model, params);

    labels.resize(segmentation->size());

    vector<size_t> sol(segmentation->size());

    for (segmentH_t i = 0; i < segmentation->size(); i++) {
        sol[i] = planeIndexMap[make_tuple(i, labels[i])];
    }

    solver.setStartingPoint(sol.begin());

    solver.infer();

    solver.arg(sol);

    for (segmentH_t i = 0; i < segmentation->size(); i++) {
        labels[i] = indexPlaneMap[make_tuple(i, sol[i])];
    }

    /*
    typedef opengm::MinSTCutBoost<size_t, float, opengm::PUSH_RELABEL> MinCutType;
    typedef opengm::GraphCut<GModel, opengm::Minimizer, MinCutType> MinGraphCut;
    // typedef opengm::AlphaBetaSwap<GModel, MinGraphCut> AlphaBetaSwap;
    typedef opengm::AlphaExpansion<GModel, MinGraphCut> MinAlphaExpansion;
    
    // AlphaBetaSwap ae(model);
    MinAlphaExpansion ae(model);

    labels.resize(segmentation->size());

    vector<size_t> sol(segmentation->size());

    for (segmentH_t i = 0; i < segmentation->size(); i++) {
        sol[i] = planeIndexMap[make_tuple(i, labels[i])];
    }

    ae.setStartingPoint(sol.begin());

    ae.infer();

    ae.arg(sol);

    for (segmentH_t i = 0; i < segmentation->size(); i++) {
        labels[i] = indexPlaneMap[make_tuple(i, sol[i])];
    }
    */
}

PlanarDepthSmoothingProblem::PlanarDepthSmoothingProblem(
        PlanarDepth* _depth,
        const Segmentation* _segmentation,
        const Connectivity* _connectivity) :
    depth(_depth),
    segmentation(_segmentation),
    connectivity(_connectivity)
{
}

void PlanarDepthSmoothingProblem::createModel(
        int numLabelsPerSeg = 30) {
    size_t numLabels = depth->getPlanes().size();

    model = unique_ptr<SegmentLabelProblem>(new SegmentLabelProblem(segmentation, numLabels, numLabelsPerSeg));

    // Store a mapping of visited segments along with their "distance" to the current
    // segment (by color and/or path length).
    set<segmentH_t> visited;

    priority_queue<tuple<float, segmentH_t>> toVisit;

    map<planeH_t, float> unaryWeights;

    auto colorCost = [](float lab1[3], float lab2[3]) -> float {
        float colorDiff = 0.0f;

        colorDiff += fabs(lab1[0] - lab2[0]);
        colorDiff += fabs(lab1[1] - lab2[1]);
        colorDiff += fabs(lab1[2] - lab2[2]);

        return colorDiff;
    };

    // Loop over all segments
    for (size_t segI = 0; segI < segmentation->size(); segI++) {
        toVisit = priority_queue<tuple<float, segmentH_t>>();
        
        visited.clear();

        unaryWeights.clear();
        
        toVisit.push(make_tuple(0.0f, segI));

        float lab[3];

        (*segmentation)[segI].avgLab(lab);
        
        while (unaryWeights.size() < numLabelsPerSeg) {
            // Pop off the index of the next segment to visit
            auto cur = toVisit.top();

            toVisit.pop();

            float colCost = get<0>(cur);
            segmentH_t curSeg = get<1>(cur);
            planeH_t planeH = depth->getSegmentPlaneMap()[curSeg];
            const Plane& plane = depth->getPlanes()[planeH];

            if (plane.isValid()) {
                float unaryCost = depth->getPlaneCostL1(segI, plane);

                unaryWeights[planeH] = unaryCost;
            }

            connectivity->forEachNeighbor(curSeg,
                    [&](size_t nI, int conn) {
                        if (visited.count(nI) == 0) {
                            visited.insert(nI);

                            float curLab[3];

                            (*segmentation)[nI].avgLab(curLab);

                            float colCost = colorCost(curLab, lab);
                            
                            toVisit.push(make_tuple(colCost, nI));
                        }
                    });
        }

        model->addUnaryFactor(segI, unaryWeights);

        float curLab[3];
        (*segmentation)[segI].avgLab(curLab);
        connectivity->forEachNeighbor(segI,
                [&](size_t nI, int conn) {
                    // Don't process pairs twice
                    if (segI < nI) {
                        float lab[3];
                        (*segmentation)[nI].avgLab(lab);

                        float colorDiff = colorCost(curLab, lab);

                        int cX1, cY1;
                        int cX2, cY2;

                        (*segmentation)[segI].getCenter(cX1, cY1);
                        (*segmentation)[nI].getCenter(cX2, cY2);

                        // Crude approximation of the point between segments
                        // segI and nI.  This enables approximation of the
                        // depth discontinuity between segments.
                        float middleX = (cX1 + cX2) / 2.0f;
                        float middleY = (cY1 + cY2) / 2.0f;

                        model->addBinaryFactor(segI, nI, 
                            [this, colorDiff, conn, middleX, middleY](planeH_t pH1, planeH_t pH2) -> float {

                                const Plane& p1 = this->depth->getPlanes()[pH1];
                                const Plane& p2 = this->depth->getPlanes()[pH2];

                                float d1 = p1.dispAt(middleX, middleY);
                                float d2 = p2.dispAt(middleX, middleY);

                                float depthDiscontinuity = fabs(d1 - d2);

                                assert(isfinite(this->smoothnessCoeff));
                                assert(isfinite(depthDiscontinuity));

                                assert(this->smoothnessCoeff >= 0.0f);
                                assert(conn >= 0.0f);
                                assert(depthDiscontinuity >= 0.0f);

                                // FIXME This pairwise energy term is stupid 
                                // (for testing submodular-energies only!)
                                if (pH1 == pH2) {
                                    return 0.5f;
                                }
                                return 1.0f;

                                // return (pH1 != pH2); // this->smoothnessCoeff * conn * depthDiscontinuity;
                        });
                    }
                });
    }
}

void PlanarDepthSmoothingProblem::solve() {
    model->solveMAP(depth->getSegmentPlaneMap());
}

