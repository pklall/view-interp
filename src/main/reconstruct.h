#pragma once

#include "common.h"
#include "cvutil/cvutil.h"

/**
 * Reconstructs 3D geometry from a sequence (chain) of images by matching
 * features in each new image to the image before it.
 */
class ChainReconstruction {
    private:
        unique_ptr<CVFeatureMatcher> prevMatcher;
        unique_ptr<CVFeatureMatcher> curMatcher;

        // matches[i] contains the set of matches (ax, ay, bx, by) such that
        // the ith image processed contains point (ax, ay) corresponding
        // to (bx, by) in the i+1 image.
        vector<vector<tuple<float, float, float, float>>> matches;

    public:
        void processNext(
                const CImg<uint8_t>& gray);

        void visualizeMatches(
                function<const CImg<uint8_t>&(int)> imgLoader) const;
};

