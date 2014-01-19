#include "segment.h"

#include "cvutil/cvutil.h"

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
    slicSuperpixels(lab, numSuperpixels, nc, segmentMap);

    superpixels = vector<Superpixel>(numSuperpixels);

    cimg_forXY(segmentMap, x, y) {
       superpixels[segmentMap(x, y)].addPixel(x, y);
    }

    for (Superpixel& s : superpixels) {
        s.compress();
    }
}

void Segmentation::renderVisualization(
        CImg<float>& result) {
    result = segmentMap.get_map(CImg<float>().lines_LUT256());
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
    planes = vector<Plane>(segmentation.size());

    // A map from y-index to (x, disparity) tuples to store
    // valid disparities for each scan-line in a superpixel.
    map<uint16_t, vector<tuple<uint16_t, float>>> xDSamples;
    
    // A map from x-index to (y, disparity) tuples to store
    // valid disparities for each vertical-line in a superpixel.
    map<uint16_t, vector<tuple<uint16_t, float>>> yDSamples;

    for (int superpixelI = 0; superpixelI < segmentation.size(); superpixelI++) {
        const auto& superpixel = segmentation[superpixelI];

        xDSamples.clear();
        yDSamples.clear();

        int numValidD = 0;

        // Iterate over all pixels within the superpixel
        for (const auto& p : superpixel.getPixels()) {
            uint16_t x = get<0>(p);
            uint16_t y = get<1>(p);

            // If this pixel has a valid disparity, add it
            if (stereo.isValidDisp(x, y)) {
                xDSamples[y].push_back(make_tuple(x, stereo.disp(x, y)));
                yDSamples[x].push_back(make_tuple(y, stereo.disp(x, y)));

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

            if (stereo.isValidDisp(x, y)) {
                float c = stereo.disp(x, y) - (cx * x + cy * y);

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
        const StereoProblem& _stereo,
        const Segmentation& _segmentation)
    : stereo(_stereo), segmentation(_segmentation) {

    fitPlanes();
}

void PlanarDepth::getDisparity(
        CImg<float>& disp) const {
    disp = CImg<float>(stereo.left.width(), stereo.right.height());

    disp = 0.0f;

    for (int superpixelI = 0; superpixelI < segmentation.size(); superpixelI++) {
        const auto& superpixel = segmentation[superpixelI];

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
