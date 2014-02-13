#include "reconstruct.h"

void ChainReconstruction::processNext(
        const CImg<uint8_t>& gray) {
    if (!curMatcher) {
        curMatcher = unique_ptr<CVFeatureMatcher>(new CVFeatureMatcher(500));
    }

    curMatcher->detectFeatures(gray);

    if (prevMatcher) {
        matches.push_back(vector<tuple<float, float, float, float>>());

        vector<tuple<float, float, float, float>>& match = matches[matches.size() - 1];

        prevMatcher->match(*curMatcher, match);
    }

    swap(curMatcher, prevMatcher);
}

void ChainReconstruction::visualizeMatches() const {
    
}

