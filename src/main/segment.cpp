#include "segment.h"

#include "cvutil/cvutil.h"

#include "localexpansion.hpp"

#include <queue>
#include <set>

Segment::Segment() {
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
    for (segmentH_t superpixelI = 0; superpixelI < superpixels.size();
            superpixelI++) {
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

    cimg_forXY(segmentMap, x, y) {
        if (validSegmentHandles.count(segmentMap(x, y)) == 0) {
            validSegmentHandles[segmentMap(x, y)] = curHandle;

            curHandle++;
        }

        superpixels[validSegmentHandles[segmentMap(x, y)]].addPixel(x, y);
    }

    superpixels.resize(curHandle);

    recomputeSegmentMap();

    for (Segment& s : superpixels) {
        s.compress();
    }

    // Compute median Lab for each superpixel
    medianLab = vector<array<float, 3>>(superpixels.size());

    vector<float> l, a, b;
    
    for (segmentH_t segH = 0; segH < superpixels.size(); segH++) {
        Segment& seg = superpixels[segH];

        l.clear();
        a.clear();
        b.clear();

        for (const auto& p : seg.getPixels()) {
            size_t x = get<0>(p);
            size_t y = get<1>(p);

            l.push_back(lab(x, y, 0, 0));
            a.push_back(lab(x, y, 0, 1));
            b.push_back(lab(x, y, 0, 2));
        }

        nth_element(&(l[0]), &(l[l.size() / 2]), &(l[l.size()]));
        nth_element(&(a[0]), &(a[a.size() / 2]), &(a[a.size()]));
        nth_element(&(b[0]), &(b[b.size() / 2]), &(b[b.size()]));

        medianLab[segH][0] = l[l.size() / 2];
        medianLab[segH][1] = a[a.size() / 2];
        medianLab[segH][2] = b[b.size() / 2];
    }
}

void Segmentation::renderVisualization(
        CImg<float>& result) const {
    result.resize(segmentMap.width(), segmentMap.height(), 1, 3, -1);

    for (segmentH_t segH = 0; segH < superpixels.size(); segH++) {
        const Segment& sp = superpixels[segH];

        const array<float, 3>& lab = medLab(segH);

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
        CImg<int16_t> _left, 
        CImg<int16_t> _right, 
        int _minDisp,
        int _maxDisp) :
    left(_left), right(_right),
    minDisp(_minDisp), maxDisp(_maxDisp) {
    assert(left.is_sameXYZC(right));

    leftLab = left.get_RGBtoLab();
    rightLab = right.get_RGBtoLab();

    disp = CImg<float>(left.width(), left.height());
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

void PlanarDepth::getPlaneCostL1(
        segmentH_t segment,
        const Plane& plane,
        float& cost,
        int& samples) const {
    if (!isInBounds(segment, plane)) {
        cost = std::numeric_limits<float>::max();

        return;
    }

    const Segment& superpixel = (*segmentation)[segment];

    cost = 0.0f;

    samples = 0;

    for (const auto& p : superpixel.getPixels()) {
        int lx = get<0>(p);
        int ly = get<1>(p);

        int rx = (int) (lx + plane.dispAt((float) lx, (float) ly) + 0.5f);
        int ry = ly;

        cimg_forC(stereo->right, c) {
            float lVal = stereo->left(lx, ly, 0, c);
            float rVal = stereo->right(rx, ry, 0, c);

            cost += fabs(lVal - rVal);
        }

        samples++;

        // if (stereo->isValidDisp(x, y)) {
            // float planeDisp = plane.dispAt((float) x, (float) y);
            // float dispSample = stereo->disp(x, y);
            // cost += fabs(planeDisp - dispSample);
            // samples++;
        // }
    }

    cost /= samples;
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

    for (segmentH_t superpixelI = 0;
            superpixelI < segmentation->size();
            superpixelI++) {
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

PlanarDepthSmoothingProblem::PlanarDepthSmoothingProblem(
        PlanarDepth* _depth,
        const StereoProblem* _stereo,
        const Segmentation* _segmentation,
        const Connectivity* _connectivity) :
    depth(_depth),
    stereo(_stereo),
    segmentation(_segmentation),
    connectivity(_connectivity) {

    createModel();
    
    unaryInlierThresh = numeric_limits<float>::max();

    binaryC0InlierThresh = numeric_limits<float>::max();
}

float PlanarDepthSmoothingProblem::UnaryCost::operator()(
        segmentH_t segH,
        planeH_t planeH) {
    const Plane& plane = self->depth->getPlanes()[planeH];

    const Segment& seg = self->segmentation->getSegments()[segH];

    int minDisp = self->stereo->minDisp;
    int maxDisp = self->stereo->maxDisp;

    float cost;
    int samples;

    self->depth->getPlaneCostL1(segH, plane, cost, samples);

    // cost = fmin(cost, self->unaryInlierThresh);

    return cost;
}

float PlanarDepthSmoothingProblem::BinaryCost::operator()(
        segmentH_t segA,
        segmentH_t segB,
        planeH_t planeA,
        planeH_t planeB) {

    const array<float, 3>& labA = (*self->segmentation).medLab(segA);
    const array<float, 3>& labB = (*self->segmentation).medLab(segB);

    float colorDiff = 0.0f;

    for (int c = 0; c < 3; c++) {
        colorDiff += fabs(labA[c] - labB[c]);
    }

    if (colorDiff > medianColorDiff) {
        return 0.0f;
    } else {
        colorDiff = 1.0f;
    }

    float depthDiscontinuity = self->pairwiseL1PlaneDist(segA, segB, planeA, planeB);

    depthDiscontinuity = fmin(depthDiscontinuity, self->binaryC0InlierThresh * 3.0f);

    // int conn = self->connectivity->getConnectivity(segA, segB);

    // FIXME this needs to be improved
    return self->smoothnessCoeff * depthDiscontinuity * colorDiff;
}

void PlanarDepthSmoothingProblem::neighborhoodGenerator(
        segmentH_t s,
        vector<segmentH_t>& neighborhood) {
    connectivity->forEachNeighbor(s,
            [&](segmentH_t segment, int conn) {
            neighborhood.push_back(segment);
            });
}

void PlanarDepthSmoothingProblem::computeUnaryCostStats() {
    /*
    int numSegs = segmentation->size();

    float costs[numSegs];

    int numValidSegs = 0;
    
    for (segmentH_t segH = 0; segH < segmentation->size(); segH++) {
        float cost;
        int samples;
        
        depth->getPlaneCostL1(segH, depth->getPlane(segH), cost, samples);
        
        // Only use valid planes and segments with at least 1 sample
        if (samples > 0 && cost < std::numeric_limits<float>::max()) {
            costs[numValidSegs] = cost;

            numValidSegs++;
        }
    }

    int medI = numValidSegs / 2;

    std::nth_element(&(costs[0]), &(costs[medI]), &(costs[numValidSegs]));

    // Estimate standard deviation via median absolute deviation
    // then get 95% conf. interval.
    unaryInlierThresh = costs[medI] * 10.0f;
    */
}

void PlanarDepthSmoothingProblem::computePairwiseCostStats() {
    vector<tuple<float, segmentH_t, int>> neighbors(8);

    for (size_t segI = 0; segI < segmentation->size(); segI++) {
        if (!depth->getPlane(segI).isValid()) {
            continue;
        }

        neighbors.clear();

        const array<float, 3>& lab = (*segmentation).medLab(segI);

        connectivity->forEachNeighbor(segI,
                [&](segmentH_t nI, int conn) {
                    const array<float, 3>& labNeighbor = (*segmentation).medLab(nI);

                    float dist = 0.0f;

                    for (int c = 0; c < 3; c++) {
                        dist += fabs(labNeighbor[c] - lab[c]);
                    }

                    if (dist < closestNeighborDist) {
                        closestNeighborDist = dist;

                        closestNeighbor = nI;

                        closestNeighborConn = conn;
                    }
                });
    }

    int medI = numValidSegs / 2;

    std::nth_element(&(c0Dist[0]), &(c0Dist[medI]), &(c0Dist[numValidSegs]));
}

void PlanarDepthSmoothingProblem::createModel() {
    UnaryCost uCost = {this};
    BinaryCost bCost = {this};

    function<void(segmentH_t, vector<planeH_t>&)> neighborGen =
        bind(&PlanarDepthSmoothingProblem::neighborhoodGenerator,
                this,
                std::placeholders::_1,
                std::placeholders::_2);

    model = unique_ptr<Solver>(new Solver(
                numSegmentsPerExpansion,
                &(depth->getSegmentPlaneMap()),
                uCost,
                bCost,
                neighborGen));
}

void PlanarDepthSmoothingProblem::computeInlierStats() {
    computeUnaryCostStats();

    computePairwiseCostStats();
}

void PlanarDepthSmoothingProblem::solve() {
    set<segmentH_t> visited;

    queue<segmentH_t> toVisit;

    set<segmentH_t> expandNodes;

    int numFlips = 0;

    // Loop over all segments and try to expand the plane
    // at that segment to adjacent nodes (by breadth-first traversal)
    for (size_t segI = 0; segI < segmentation->size(); segI++) {
        if (!depth->getPlane(segI).isValid()) {
            continue;
        }

        toVisit = queue<segmentH_t>();
        
        visited.clear();

        toVisit.push(segI);

        expandNodes.clear();

        while (expandNodes.size() < numSegmentsPerExpansion) {
            // Pop off the index of the next segment to visit
            segmentH_t curSeg = toVisit.front();

            toVisit.pop();

            expandNodes.insert(curSeg);

            connectivity->forEachNeighbor(curSeg,
                    [&](segmentH_t nI, int conn) {
                        if (visited.count(nI) == 0) {
                            visited.insert(nI);

                            toVisit.push(nI);
                        }
                    });
        }

        numFlips += model->tryExpand(expandNodes, depth->getSegmentPlaneMap()[segI]);
    }

    printf("Flips = %d\n", numFlips);
}

