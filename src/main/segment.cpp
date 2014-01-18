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
