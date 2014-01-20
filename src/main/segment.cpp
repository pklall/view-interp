#include "segment.h"

#include "cvutil/cvutil.h"

Superpixel::Superpixel() : totalLab{0.0f, 0.0f, 0.0f} {
    minX = std::numeric_limits<uint16_t>::max();
    minY = std::numeric_limits<uint16_t>::max();

    maxX = std::numeric_limits<uint16_t>::min();
    maxY = std::numeric_limits<uint16_t>::min();
}

void Connectivity::increment(
        size_t a,
        size_t b) {
    connectivity[a][b]++;
    connectivity[b][a]++;
}

int Connectivity::getConnectivity(
        size_t a,
        size_t b) const {
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
    for (int superpixelI = 0; superpixelI < superpixels.size(); superpixelI++) {
        const auto& pixels = superpixels[superpixelI].getPixels();

        for (const auto& p : pixels) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            segmentMap(x, y) = superpixelI;
        }
    }
}

void Segmentation::createSlicSuperpixels(
        const CImg<float>& lab,
        int numSuperpixels,
        int nc) {
    assert(lab.spectrum() == 3);

    slicSuperpixels(lab, numSuperpixels, nc, segmentMap);

    superpixels = vector<Superpixel>(numSuperpixels);

    float labTmp[3];
    cimg_forXY(segmentMap, x, y) {
        cimg_forC(lab, c) {
            labTmp[c] = lab(x, y, 0, c);
        }

        superpixels[segmentMap(x, y)].addPixel(x, y, labTmp);
    }

    for (Superpixel& s : superpixels) {
        s.compress();
    }
}

void Segmentation::renderVisualization(
        CImg<float>& result) const {
    result.resize(segmentMap.width(), segmentMap.height(), 1, 3, -1);

    for (const Superpixel& sp : superpixels) {
        float lab[3];

        sp.avgLab(lab);

        cimg_forC(result, c) {
            for (const auto& coord : sp.getPixels()) {
                uint16_t x = get<0>(coord);
                uint16_t y = get<1>(coord);

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
            size_t segCur = segmentMap(x, y);
            size_t segLeft = segmentMap(x - 1, y);
            size_t segTop = segmentMap(x, y - 1);

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
        const map<uint16_t, vector<tuple<uint16_t, float>>>& dSamples,
        float& result) const {
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
        const vector<tuple<uint16_t, float>>& samples = samplesPair.second;

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
    // dtSamples.blur(min(1.0f, dtSamples.size() / 16.0f));

    result = dtSamples(dtSamples.size() / 2);

    return true;
}

void PlanarDepth::fitPlanes() {
    // Create a plane for each superpixel
    planes = vector<Plane>(segmentation->size());

    // A map from y-index to (x, disparity) tuples to store
    // valid disparities for each scan-line in a superpixel.
    map<uint16_t, vector<tuple<uint16_t, float>>> xDSamples;
    
    // A map from x-index to (y, disparity) tuples to store
    // valid disparities for each vertical-line in a superpixel.
    map<uint16_t, vector<tuple<uint16_t, float>>> yDSamples;

    for (int superpixelI = 0; superpixelI < segmentation->size(); superpixelI++) {
        const auto& superpixel = (*segmentation)[superpixelI];

        xDSamples.clear();
        yDSamples.clear();

        int numValidD = 0;

        // Iterate over all pixels within the superpixel
        for (const auto& p : superpixel.getPixels()) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            // If this pixel has a valid disparity, add it
            if (stereo->isValidDisp(x, y)) {
                xDSamples[y].push_back(make_tuple(x, stereo->disp(x, y)));
                yDSamples[x].push_back(make_tuple(y, stereo->disp(x, y)));

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
        for (const auto& p : superpixel.getPixels()) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

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

        planes[superpixelI] = Plane(cx, cy, c);
    }
}


PlanarDepth::PlanarDepth(
        const StereoProblem* _stereo,
        const Segmentation* _segmentation)
    : stereo(_stereo), segmentation(_segmentation) {
    fitPlanes();
}

void PlanarDepth::getDisparity(
        CImg<float>& disp) const {
    disp = CImg<float>(stereo->left.width(), stereo->right.height());

    disp = 0.0f;

    for (int superpixelI = 0; superpixelI < segmentation->size(); superpixelI++) {
        const auto& superpixel = (*segmentation)[superpixelI];

        const Plane& plane = planes[superpixelI];

        if (plane.isValid()) {
            for (const auto& p : superpixel.getPixels()) {
                uint16_t x = get<0>(p);
                uint16_t y = get<1>(p);

                disp(x, y) = plane.dispAt(x, y);
            }
        }
    }
}

void PlanarDepth::renderInterpolated(
        float t,
        CImg<float>& result) {
    vector<size_t> segmentIndices(segmentation->size());

    for (size_t i = 0; i < segmentation->size(); i++) {
        segmentIndices[i] = i;
    }

    // Sort segments (by index) according to depth at center
    // for back-to-front rendering (Painter's Algo.)
    std::sort(segmentIndices.begin(), segmentIndices.end(), [&](size_t a, size_t b) {
            int aX, aY, bX, bY;
            (*segmentation)[a].getCenter(aX, aY);
            (*segmentation)[b].getCenter(bX, bY);
            return planes[a].dispAt(aX, aY) > planes[b].dispAt(bX, bY);
            });

    // Add an alpha channel to the result
    result.resize(stereo->left.width(), stereo->left.height(), 1,
            stereo->left.spectrum(), -1);

    result = 0.0f;
    
    for (size_t segI : segmentIndices) {
        const Superpixel& superpixel = (*segmentation)[segI];

        Plane& plane = planes[segI];

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
        size_t _numLabels) :
    segmentation(_segmentation), numLabels(_numLabels) {

    Space space(segmentation->size(), numLabels);

    model = GModel(space);
}

void SegmentLabelProblem::addUnaryFactor(
        size_t segment,
        function<float(size_t)> labelFactor) {
    size_t shape[] = {(size_t) numLabels};

    opengm::ExplicitFunction<float> dataTerm(shape, shape + 1);

    for (size_t labelI = 0; labelI < numLabels; labelI++) {
        dataTerm(labelI) = labelFactor(labelI);
    }

    GModel::FunctionIdentifier fid = model.addFunction(dataTerm);

    size_t vars[] = {(size_t) numLabels};
    model.addFactor(fid, begin(vars), end(vars));
}

void SegmentLabelProblem::addBinaryFactor(
        size_t segment1,
        size_t segment2,
        float factor) {
}

void SegmentLabelProblem::solveMAP() {
    typedef opengm::MinSTCutBoost<size_t, float, opengm::PUSH_RELABEL> MinCutType;
    typedef opengm::GraphCut<GModel, opengm::Minimizer, MinCutType> MinGraphCut;
    typedef opengm::AlphaExpansion<GModel, MinGraphCut> MinAlphaExpansion;
    
    MinAlphaExpansion ae(model);
    
    ae.infer();

    labels = vector<size_t>(segmentation->size());
    ae.arg(labels);
}

