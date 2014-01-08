#include "common.h"

#include "pmstereo.h"

#include <omp.h>

/**
 * Implementation of generalized PatchMatch.
 *
 * Note that field, totCost, and fieldSorted, consistute a
 * structure-of-arrays such that values in each array with the same
 * (x, y, k, _) correspond to properties of the same "particle" for that (x, y)
 * coordinate. Hence, the width, height, and depth of each of these must be
 * identical.
 *
 * \param field The K best values at each pixel, where each value is a
 *              vector of arbitrary length.
 * \param totCost The total cost of each of the K best values at each pixel
 * \param fieldSorted The index of each value in field, sorted by totCost,
 *                    such that field(x, y, fieldSorted(0), _) is the optimal
 *                    value (with lowest totCost).
 * \param propagation Given an (x, y) coordinate and the index into a logical
 *                    list of propagation sources (e.g. top, bottom, left,
 *                    right, random...), generates a candidate value and
 *                    returns true.  Returns false if the index is out
 *                    of bounds.
 * \param unaryCost Computes the cost of the given (x, y, value) triple
 * \param increment Determines the step size to use when considering (x, y)
 *                  values.  This enables coarse-to-fine processing.
 */
inline void patchMatch(
        CImg<float>& field,
        CImg<float>& totCost,
        CImg<int>& fieldSorted,
        function<bool(int, int, int, float[])> getCandidateValue,
        function<float(int, int, float[])> unaryCost,
        bool reverse,
        int increment = 1) {
    assert(field.is_sameXYZ(totCost));
    assert(field.is_sameXYZ(fieldSorted));

    // The number of particles
    int K = field.depth();

    // The dimension of the field
    int valSize = field.spectrum();

    assert(totCost.spectrum() == 1);
    assert(fieldSorted.spectrum() == 1);

#pragma omp parallel
    {
        int xStart = 0;
        int yStart = 0;
        int inc = increment;
        if (reverse) {
            xStart = field.width() - 1;
            yStart = field.height() - 1;
            inc = -1 * increment;
        }

        // FIXME this probably fails if inc isn't of magnitude 1
        int numThreads = omp_get_num_threads();
        int stride = (field.height() + numThreads - 1) / numThreads;
        int yLimit = yStart + inc * (omp_get_thread_num() + 1) * stride;
        yStart += inc * omp_get_thread_num() * stride;

#pragma omp critical
        {
        cout << "Thread #" << omp_get_thread_num() << " yStart = " << yStart
            << " yLimit = " << yLimit << " stride = " << stride << endl;
        }
        for (int y = yStart; y >= 0 && y < field.height() && y != yLimit; y += inc) {
            for (int x = xStart; x >= 0 && x < field.width(); x += inc) {
                // Space to store the candidate field value
                float cVal[valSize];

                // Loop over all candidate new values, based on
                // the propagation function
                for (int pNum = 0; getCandidateValue(x, y, pNum, cVal); pNum++) {
                    if (cVal[0] == std::numeric_limits<float>::max()) {
                        continue;
                    }

                    float totalCost = unaryCost(x, y, cVal);

                    // Find the index of the first particle with a greater cost
                    // in the sorted list.
                    int index = -1;
                    for (int i = 0; i < K; i++) {
                        if (totCost(x, y, fieldSorted(x, y, i)) > totalCost) {
                            index = i;
                            break;
                        }
                    }

                    // If this new particle is good, insert it into our list of
                    // optimal particles.
                    if (index != -1 ) {

                    // numSearchWinners[pNum]++;

                    // The "raw index" is the index into field, totCost, ...
                    // which will store this particle.
                    // Since we're inserting this new particle, the last
                    // particle in the sorted list will be eliminated.  Thus
                    // we'll use it's now-unused "raw" slot to store the new
                    // particle.
                    // This indirection is useful since we avoid moving
                    // lots of data around, and can instead simply shift
                    // down the indices in the sorted list.
                    int rawIndex = fieldSorted(x, y, K - 1);

                    cimg_forC(field, c) {
                        field(x, y, rawIndex, c) = cVal[c];
                    }

                    totCost(x, y, rawIndex) = totalCost;

                    // Pull back all inferior particles to make room
                    for (int i = K - 1; i >= index + 1; i--) {
                        fieldSorted(x, y, i) = fieldSorted(x, y, i - 1);
                    }

                    fieldSorted(x, y, index) = rawIndex;
                    }
                }
            }
        }
    }
}

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
                        float diff = abs(lab1(x + sx, y + sy, z, c) -
                            lab2(x + dx, y + dy, z, c));

                        diff = min(diff, maxDist);

                        float gradDiff = abs(grad1(x + sx, y + sy, z, c) -
                            grad2(x + dx, y + dy, z, c));

                        gradDiff = min(gradDiff, maxGradDist);

                        totalWeight += weight;

                        ssd += (diff * 0.1f + gradDiff * 0.9f) * weight;
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

            if (!(newX < 0 || newX >= fieldLeft.width() ||
                    newY < 0 || newY >= fieldLeft.height())) {
                float newDisp = fieldLeft(newX, newY, z, 0);
                value[0] = newDisp;
            } else {
                value[0] = std::numeric_limits<float>::max();
            }
        } else if (i < 4 * K) {
            // int newX = x, newY = y;

            int z = i - 3 * K;

            value[0] = (*fieldRightRev)(x, y, z, 0);
            
            /*
            newX += fieldLeft(x, y, z, 0);

            // Quantize the grid based on the current increment
            newX = (newX / increment) * increment;

            if (!(newX < 0 || newX >= fieldRight.width() ||
                    newY < 0 || newY >= fieldRight.height())) {
                // TODO try different z values here too?
                int z2 = 0;

                value[0] = -fieldRight(newX, newY, z2, 0);
            } else {
                value[0] = std::numeric_limits<float>::max();
            }
            */
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

        patchMatch(fieldLeft, distLeft, sortedLeft, candidateLeft, patchDistLeft,
                iter % 2 == 0, increment);

        reverseTranslationalField(fieldLeft, fieldLeftRev);

        patchMatch(fieldRight, distRight, sortedRight, candidateRight, patchDistRight,
                iter % 2 == 0, increment);
    }
}

inline function<float(int, int, float[])> affinePatchDist(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        int wndSize,
        float colorSigma = 10.0f,
        float maxDist = 10.0f) {
    auto dist = [&lab1, &lab2, wndSize, colorSigma, maxDist]
        (int sx, int sy, float* value) -> float {
            float ssd = 0.0f;
            float totalWeight = 0.0f;

            for (int y = -wndSize / 2; y <= wndSize / 2; y++) {
                for (int x = -wndSize / 2; x <= wndSize / 2; x++) {
                    int srcX = x + sx;
                    int srcY = y + sy;

                    if (srcX >= 0 && srcX < lab1.width() &&
                            srcY >= 0 && srcY < lab1.height()) {

                        float dstX = srcX +
                            value[0] + value[1] * srcX + value[2] * srcY;
                        float dstY = srcY;

                        float lab1Diff = 0.0f;
                        cimg_forZC(lab1, z, c) {
                            float lDiff = lab1(srcX, srcY, z, c) -
                                lab1(sx, sy, z, c);
                            lab1Diff += abs(lDiff);
                        }
                        float weight = exp(-(lab1Diff) / colorSigma);

                        if (dstX >= 0 && dstX < lab2.width() &&
                                dstY >= 0 && dstY < lab2.height()) {
                            cimg_forZC(lab1, z, c) {
                                float diff = lab1(srcX, srcY, z, c) -
                                    lab2.linear_atXYZC(dstX, dstY, z, c);

                                diff = min(diff, maxDist);

                                ssd += abs(diff) * weight;
                                totalWeight += weight;
                            }
                        }
                    }
                }
            }

            if (totalWeight == 0.0f) {
                return std::numeric_limits<float>::max();
            }

            return ssd / totalWeight;
        };
    return dist;
}
