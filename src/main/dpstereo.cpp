#include "dpstereo.h"

DPStereo::DPStereo(
        const Segmentation* _segmentation,
        int _smallDisp,
        float _costLargeDispSegEdge,
        float _costLargeDisp) :
    segmentation(_segmentation),
    smallDisp(_smallDisp),
    costLargeDispSegEdge(_costLargeDispSegEdge),
    costLargeDisp(_costLargeDisp) {
}

void DPStereo::computeStereo(
        StereoProblem& problem) {
    const CImg<uint16_t>& left = problem.left;
    const CImg<uint16_t>& right = problem.right;
    CImg<float>& disp = problem.disp;

    int minX = max(0, 0 - problem.minDisp);
    int maxX = min(left.width() - 1, left.width() - 1 - problem.maxDisp);

    int numX = maxX - minX + 1;

    int minDisp = problem.minDisp;
    int numD = problem.maxDisp - problem.minDisp + 1;

    CImg<uint16_t> costVol(numD);

    CImg<int16_t> dpPredD(numD, numX);

    CImg<uint16_t> dpMinCost(numD, numX);

    disp = 0.0f;

    for (int y = 0; y < disp.height(); y++) {
        printf("processing y = %d\n", y);
        int optimalDI;
        // TODO cache block and vectorize
        for (int xI = 0; xI < numX; xI++) {
            int x = xI + minX;

            for (int dI = 0; dI < numD; dI++) {
                int d = dI + minDisp;

                costVol(dI) = 0;

                cimg_forC(left, c) {
                    costVol(dI) += abs(
                            (int16_t) left(x, y, c) -
                            (int16_t) right(x + d, y, c));
                }
            }

            if (xI == 0) {
                optimalDI = 0;
                int optimalDCost = std::numeric_limits<int>::max();

                for (int dI = 0; dI < numD; dI++) {
                    dpPredD(dI, xI) = 0;
                    dpMinCost(dI, xI) = costVol(dI);

                    if (dpMinCost(dI, xI) < optimalDCost) {
                        optimalDCost = dpMinCost(dI, xI);

                        optimalDI = dI;
                    }
                }
            } else {
                // Naively iterating for d in [0, numD] is *really* slow.
                // So instead, only iterate over
                // [dI - smallDisp, dI + smallDisp], clamped to [minDisp, maxDisp].
                // Handle large disparity jumps by tracking the lowest cost
                // in for the previous x-value, and jump to that (this works
                // for potts-like pairwise costs).
                bool isSegmentEdge = (*segmentation)(x, y) != (*segmentation)(x - 1, y);

                int nextOptimalDI = 0;
                int optimalDCost = std::numeric_limits<int>::max();

                // This is *really* slow!
                for (int dI = 0; dI < numD; dI++) {
                    int minDI = max(0, dI - smallDisp);
                    int maxDI = max(numD - 1, dI + smallDisp);

                    int16_t curUnaryCost = costVol(dI);

                    dpPredD(dI, xI) = optimalDI;
                    dpMinCost(dI, xI) = curUnaryCost +
                        dpMinCost(optimalDI, xI - 1);

                    if (!isSegmentEdge) {
                        dpMinCost(dI, xI) += costLargeDispSegEdge;
                    } else {
                        dpMinCost(dI, xI) += costLargeDisp;
                    }

                    for (int dpI = minDI; dpI <= maxDI; dpI++) {
                        int newCost = 
                            curUnaryCost +
                            dpMinCost(dpI, xI - 1) +
                            abs(dpI - dI);
                        
                        if (newCost < dpMinCost(dI, xI)) {
                            dpMinCost(dI, xI) = (int16_t) newCost;

                            dpPredD(dI, xI) = dpI;
                        }
                    }

                    if (dpMinCost(dI, xI) < optimalDCost) {
                        optimalDCost = dpMinCost(dI, xI);

                        nextOptimalDI = dI;
                    }
                }

                optimalDI = nextOptimalDI;
            }

            // Read off the dp solution
            if (xI == (maxX - minX)) {
                // Find the optimal disparity for the right-most x-value
                int minCost = std::numeric_limits<int>::max();
                int bestDI = 0;

                for (int dI = 0; dI < numD; dI++) {
                    if (dpMinCost(dI, xI) < minCost) {
                        minCost = dpMinCost(dI, xI);

                        bestDI = dI;
                    }
                }

                int curDI = bestDI;
                for (int xbI = xI; xbI >= 0; xbI--) {
                    disp(xbI + minX, y) = (float) curDI + (float) minDisp;

                    curDI = dpPredD(curDI, xbI);
                }
            }
        }
    }

    disp.display();
}
