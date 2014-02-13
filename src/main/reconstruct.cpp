#include "reconstruct.h"

void ChainReconstruction::processNext(
        const CImg<uint8_t>& gray) {
    if (!curMatcher) {
        curMatcher = unique_ptr<CVFeatureMatcher>(new CVFeatureMatcher(1024));
    }

    curMatcher->detectFeatures(gray);

    if (prevMatcher) {
        matches.push_back(vector<tuple<float, float, float, float>>());

        vector<tuple<float, float, float, float>>& match = matches[matches.size() - 1];

        prevMatcher->match(*curMatcher, match);
    }

    swap(curMatcher, prevMatcher);
}

void ChainReconstruction::visualizeMatches(
        function<const CImg<uint8_t>&(int)> imgLoader) const {
    const CImg<uint8_t>* prevImg = NULL;
    const CImg<uint8_t>* curImg = NULL;

    prevImg = &imgLoader(0);

    for (int i = 0; i < matches.size(); i++) {
        curImg = &imgLoader(i + 1);
        
        assert(prevImg != NULL);
        assert(curImg != NULL);

        CImg<uint8_t> annotation(
                prevImg->width() + curImg->width(),
                max(prevImg->height(), curImg->height()),
                1, 1);

        annotation.draw_image(0, 0, 0, 0, *prevImg);

        annotation.draw_image(prevImg->width(), 0, 0, 0, *curImg);

        annotation.resize(-100, -100, -100, 3);

        CImg<uint8_t> col = CImg<uint8_t>::lines_LUT256();

        int matchI = 0;
        for (const auto& match : matches[i]) {
            // Only draw the best 64 matches
            if (matchI > 64) {
                break;
            }

            float prevX = get<0>(match);
            float prevY = get<1>(match);

            float curX = get<2>(match);
            float curY = get<3>(match);

            uint8_t color[3];

            for (int c = 0; c < 3; c++) {
                color[c] = col(matchI % 256, c);
            }

            annotation.draw_line(
                    (int) (prevX + 0.5f),
                    (int) (prevY + 0.5f),
                    (int) (curX + 0.5f) + prevImg->width(),
                    (int) (curY + 0.5f),
                    color);

            matchI++;
        }

        annotation.display();
        
        swap(prevImg, curImg);
    }
}

