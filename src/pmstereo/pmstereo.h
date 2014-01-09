#pragma once

#include "common.h"

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
        int increment);

void translationalConsistency(
        const CImg<float>& left,
        const CImg<float>& right,
        CImg<bool>& consistent);
