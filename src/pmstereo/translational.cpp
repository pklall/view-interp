#include "common.h"

#include "pmstereo.h"

inline function<float(int, int, float[])> translationalPatchDist(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        const CImg<float>& grad1,
        const CImg<float>& grad2,
        int wndSize,
        float colorSigma = 10.0f,
        float maxDist = 10.0f,
        float maxGradDist = 2.0f) {
    auto dist = [=] (int sx, int sy, float* value) -> float {
            int dx = sx + (int) value[0];
            int dy = sy;

            if ( 
                    sx - wndSize / 2 < 0 || sx + wndSize / 2 >= lab1.width() ||
                    sy - wndSize / 2 < 0 || sy + wndSize / 2 >= lab1.height()||
                    dx - wndSize / 2 < 0 || dx + wndSize / 2 >= lab2.width() ||
                    dy - wndSize / 2 < 0 || dy + wndSize / 2 >= lab2.height()) {
                return numeric_limits<float>::infinity();
            }

            int minX = -wndSize / 2;
            int maxX =  wndSize / 2;
            int minY = -wndSize / 2;
            int maxY =  wndSize / 2;

            float totalWeight = 0.0f;
            float ssd = 0.0f;
            for (int y = minY; y <= maxY; y++) {
                for (int x = minX; x <= maxX; x++) {
                    // Weight pixels with a bilateral-esque filter
                    float lab1Diff = 0.0f;

                    cimg_forZC(lab1, z, c) {
                        float lDiff = lab1(x + sx, y + sy, z, c) -
                            lab1(sx, sy, z, c);
                        lab1Diff += abs(lDiff);
                    }

                    float weight = exp(-lab1Diff / colorSigma);

                    cimg_forZC(lab1, z, c) {

                        // diff = min(diff, maxDist);

                        // float gradDiff = abs(grad1(x + sx, y + sy, z, c) -
                            // grad2(x + dx, y + dy, z, c));

                        // gradDiff = min(gradDiff, maxGradDist);

                        float diff = lab1(x + sx, y + sy, z, c) -
                            lab2(x + dx, y + dy, z, c);

                        totalWeight += weight;

                        ssd += diff * diff * weight;// (diff * 0.1f + gradDiff * 0.9f) * weight;
                    }
                }
            }

            return ssd / totalWeight;
        };
    return dist;
}

/**
 * Creates a function for generating candidate translational disparities.
 */
inline function<bool(int, int, int, float[])> translationalCandidateGenerator(
        const CImg<float>& fieldLeft,
        const CImg<float>& fieldRight,
        const CImg<float>* fieldRightRev,
        int* iterationCounter,
        float randomSearchFactor = 1.0f,
        int increment = 1) {
    assert(fieldLeft.depth() == fieldRight.depth());

    int K = fieldLeft.depth();

    return [=](int x, int y, int i, float* value) mutable {
        if (i >= 4 * K) {
            return false;
        }

        if (i < K) {
            // Random sample
            float searchWndRadiusFactor = randomSearchFactor / pow(2.0f, *iterationCounter);

            int width = fieldRight.width();

            float searchWndWidth  = searchWndRadiusFactor * width;

            int z = i;

            float minSearchWndX = x + fieldLeft(x, y, z, 0) - searchWndWidth / 2.0f;

            float maxSearchWndX = x + fieldLeft(x, y, z, 0) + searchWndWidth / 2.0f;

            minSearchWndX = max(0.0f, minSearchWndX);

            maxSearchWndX = min((float) width, maxSearchWndX);

            // Randomly choose an absolute coordinate
            int randX = (int) (cimg::rand() * (maxSearchWndX - minSearchWndX) + minSearchWndX);

            // Store the relative disparity
            value[0] = randX - x;
        } else if (i < 2 * K || i < 3 * K) {
            // Propagate from neighbors on the same view
            int newX = x, newY = y;

            int z = 0;
            if (i < 2 * K) {
                z = i - K;
                // propagate left/right
                if (*iterationCounter % 2 == 0) {
                    newX += increment;
                } else {
                    newX -= increment;
                }
            } else {
                z = i - 2 * K;
                // propagate up/down
                if (*iterationCounter % 2 == 0) {
                    newY += increment;
                } else {
                    newY -= increment;
                }
            }

            if (newX < 0 || newX >= fieldLeft.width() ||
                    newY < 0 || newY >= fieldLeft.height()) {
                value[0] = std::numeric_limits<float>::max();
            } else {
                float newDisp = fieldLeft(newX, newY, z, 0);
                value[0] = newDisp;
            }
        } else if (i < 4 * K) {
            int z = i - 3 * K;

            value[0] = (*fieldRightRev)(x, y, z, 0);
        }

        return true;
    };
}

inline void reverseTranslationalField(
        const CImg<float>& field,
        CImg<float>& fieldRev) {
    fieldRev = std::numeric_limits<float>::max();

    cimg_forXYZ(field, x, y, z) {
        int rx = x + field(x, y, z, 0);
        int ry = y;

        if (rx >= 0 && rx < fieldRev.width() &&
                ry >= 0 && ry < fieldRev.height()) {
            fieldRev(rx, ry, z, 0) = -field(x, y, z, 0);
        }
    }
}

/**
 * Uses PatchMatch to solve for translational correspondence (without slanted
 * support windows) with integer disparity precision.
 */
void patchMatchTranslationalCorrespondence(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        const CImg<float>& grad1,
        const CImg<float>& grad2,
        CImg<float>& fieldLeft,
        CImg<float>& fieldRight,
        CImg<float>& distLeft,
        CImg<float>& distRight,
        CImg<int>& sortedLeft,
        CImg<int>& sortedRight,
        int wndSize,
        int iterations,
        float randomSearchFactor,
        int increment) {
    int iter = 0;

    CImg<float> fieldLeftRev(fieldLeft);
    CImg<float> fieldRightRev(fieldRight);

    auto candidateLeft = translationalCandidateGenerator(
            fieldLeft, fieldRight, &fieldRightRev,
            &iter, randomSearchFactor, increment);

    auto candidateRight = translationalCandidateGenerator(
            fieldRight, fieldLeft, &fieldLeftRev,
            &iter, randomSearchFactor, increment);

    auto patchDistLeft = translationalPatchDist(lab1, lab2,
            grad1, grad2, wndSize);

    auto patchDistRight = translationalPatchDist(lab2, lab1,
            grad1, grad2, wndSize);

    for (; iter < iterations; iter++) {
        reverseTranslationalField(fieldRight, fieldRightRev);

        patchMatch(fieldLeft, distLeft, sortedLeft,
                candidateLeft, patchDistLeft, iter % 2 == 0, increment);

        reverseTranslationalField(fieldLeft, fieldLeftRev);

        patchMatch(fieldRight, distRight, sortedRight,
                candidateRight, patchDistRight, iter % 2 == 0, increment);
    }
}


void translationalConsistency(
        const CImg<float>& fieldLeft,
        const CImg<float>& fieldRight,
        CImg<bool>& consistent) {
    cimg_forXY(fieldLeft, x, y) {
        int rx = x + fieldLeft(x, y, 0, 0);
        int ry = y;
        consistent(x, y) = false;
        if (rx >= 0 && rx < fieldRight.width() &&
                ry >= 0 && ry < fieldRight.height()) {
            if ((fieldRight(rx, ry) + rx - x) < 2.0f) {
                consistent(x, y) = true;
            }
        }
    }
}
