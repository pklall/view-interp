#include "common.h"

#include "files.h"

using namespace std;

using namespace cimg_library;

#define dbgOut std::cout

void featureMatchCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& matches);

void visualizeCorrespondence(
        const CImg<float>& from,
        const CImg<int>& corr,
        const CImg<float>& to) {
    CImgDisplay fromDisp(from);
    CImgDisplay corrDisp(corr.get_normalize(0, 255).get_equalize(0, 255));
    CImgDisplay toDisp(to);

    CImg<float> toAnnotated = to;

    while (!fromDisp.is_closed()) {
        fromDisp.wait();

        // mouse coordinates
        int mx = fromDisp.mouse_x();
        int my = fromDisp.mouse_y();

        toAnnotated = to;
        if (mx >= 0 && mx < corr.width() &&
                my >= 0 && my < corr.height()) {
            int corrX = mx + corr(mx, my, 0);
            int corrY = my;

            float circleColor[3] = {255.0f, 0.0f, 0.0f};
            toAnnotated.draw_circle(corrX, corrY, 5, circleColor, 1.0f, 0);
        }

        toDisp.display(toAnnotated);
    }
}

void visualizeVectorImg(const CImg<float>& vec) {
    CImg<float> proj(vec.get_channel(0));

    CImgDisplay projDisp(proj);

    while (!projDisp.is_closed()) {
        projDisp.wait();
        // mouse coordinates
        int mx = projDisp.mouse_x();
        int my = projDisp.mouse_y();

        // mouse vector relative to center
        float mvx = mx - (projDisp.width()/2.0f);
        float mvy = my - (projDisp.height()/2.0f);
        float mvnorm = norm(mvx, mvy);

        // normalized mouse...
        float nmvx = mvx / mvnorm;
        float nmvy = mvy / mvnorm;

        // Update proj with the magnitude of the gradient onto
        // the mouse vector
        cimg_forXYZC(proj, x, y, z, c) {
            float xv = vec(x, y, 0);
            float yv = vec(x, y, 1);
            float px = xv * nmvx + yv * nmvy;
            proj(x, y, z, c) = px;
        }

        proj.display(projDisp);
    }
}

void displayClickable(CImg<float>& img,
        std::function<void(int, int)> onClick) {
    CImgDisplay disp(img);
    while (!disp.is_closed()) {
        disp.wait();
        if (!(disp.button() && disp.mouse_y()>=0)) {
            continue;
        }

        onClick(disp.mouse_x(), disp.mouse_y());

        disp.display(img);
    }
}

/**
 * Implementation of generalized PatchMatch belief propagation.
 *
 * Note that field, totCost, fieldSorted, dist, and msg consistute a
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
 * \param dist The unary cost of each of the K best values.
 * \param msg The cost of each of the K best values, as reported by messages
 *            from the top, bottom, left, and right neighbors, respectively.
 * \param propagation Given an (x, y) coordinate and the index into a logical
 *                    list of propagation sources (e.g. top, bottom, left,
 *                    right, random...), generates a candidate value and
 *                    returns true.  Returns false if the index is out
 *                    of bounds.
 * \param unaryCost Computes the cost of the given (x, y, value) triple
 * \param binaryCost Computes the cost of a pair of (x, y, value) triples
 * \param increment Determines the step size to use when considering (x, y)
 *                  values.  This enables coarse-to-fine processing.
 */
inline void patchMatchBeliefPropagation(
        CImg<float>& field,
        CImg<float>& totCost,
        CImg<int>& fieldSorted,
        CImg<float>& dist,
        CImg<float>& msg,
        function<bool(int, int, int, float[])> getCandidateValue,
        function<float(int, int, float[])> unaryCost,
        function<float(int, int, float[], int, int, float[])> binaryCost,
        int increment = 1,
        bool purePM = false) {
    assert(field.is_sameXYZ(totCost));
    assert(field.is_sameXYZ(fieldSorted));
    assert(field.is_sameXYZ(dist));
    assert(field.is_sameXYZ(msg));

    int K = field.depth();

    // The size of each value in the field
    int valSize = field.spectrum();
    assert(totCost.spectrum() == 1);
    assert(fieldSorted.spectrum() == 1);
    assert(msg.spectrum() == 4);
    assert(dist.spectrum() == 1);

    const int neighbors[][2] = {
        {0, -1}, // top
        {0, 1},  // bottom
        {-1, 0}, // left
        {1, 0}   // right
    };

    const int inverseNeighbors[] = {1, 0, 3, 2};

    // TODO Cache block
    // FIXME This has race conditions. Please fix!
#pragma omp parallel for
    for (int y = 0; y < field.height(); y += increment) {
        for (int x = 0; x < field.width(); x += increment) {
            // Space to store the candidate field value
            float cVal[valSize];

            // Loop over all candidate new values, based on
            // the propagation function
            for (int pNum = 0; getCandidateValue(x, y, pNum, cVal); pNum++) {
                // TODO There's a possible early-exit here when the total cost
                //      of cVal is greater than the lowest-ranked particle
                //      with cost totCost(x, y, fieldSorted(K - 1));
                float uCost = unaryCost(x, y, cVal);

                float msgs[4];
                float msgTotal = 0;

                if (!purePM) {
                    // Compute new messages from neighbors
                    for (int i = 0; i < 4; i++) {
                        int adjX = x + neighbors[i][0] * increment;
                        int adjY = y + neighbors[i][1] * increment;

                        if (adjX >= field.width() || adjX < 0 ||
                                adjY >= field.height() || adjY < 0) {
                            // Ignore this neighbor if it's out-of-bounds.
                            msgs[i] = 0;
                            continue;
                        }

                        msgs[i] = std::numeric_limits<float>::max();

                        // The message from a neighbor is minimized
                        // over all particles for that neighbor
                        cimg_forZ(field, z) {
                            float adjVal[valSize];

                            cimg_forC(field, c) {
                                adjVal[c] = field(adjX, adjY, z, c);
                            }

                            float psi = binaryCost(x, y, cVal, adjX, adjY, adjVal);

                            // Candidate message, we use the min-such value
                            // over all particles at (adjX, adjY).
                            float cMsg =
                                psi +
                                dist(adjX, adjY, z) -
                                msg(adjX, adjY, z, inverseNeighbors[i]);

                            msgs[i] = min(msgs[i], cMsg);
                        }

                        msgTotal += msgs[i];
                    }
                }

                float totalCost = msgTotal + uCost;

                // Find the index of the first particle with a greater cost
                // in the sorted list.
                int index = -1;
                for (int i = 0; i < K; i++) {
                    if (totCost(x, y, fieldSorted(x, y, i)) > totalCost) {
                        index = i;
                        break;
                    }
                }

                // If this new particle sucks, continue because there's nothing
                // left to do with it.
                if (index == -1 ) {
                    continue;
                }

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

                // Pull back all inferior particles to make room
                for (int i = K - 1; i >= index + 1; i--) {
                    fieldSorted(x, y, i) = fieldSorted(x, y, i - 1);
                }

                fieldSorted(x, y, index) = rawIndex;

                cimg_forC(field, c) {
                    field(x, y, rawIndex, c) = cVal[c];
                }

                totCost(x, y, rawIndex) = totalCost;

                dist(x, y, rawIndex) = uCost;

                if (!purePM) {
                    for (int i = 0; i < 4; i++) {
                        msg(x, y, rawIndex, i) = msgs[i];
                    }
                }
            }
        }
    }
}

inline function<float(int, int, float[])> translationalPatchDist(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        int wndSize,
        float colorSigma = 30.0f,
        float maxDist = 10.0f) {
    auto dist = [=]
        (int sx, int sy, float* value) -> float {
            int dx = sx + (int) value[0];
            int dy = sy;

            if ( 
                    sx - wndSize / 2 < 0 || sx + wndSize / 2 >= lab1.width() ||
                    sy - wndSize / 2 < 0 || sy + wndSize / 2 >= lab1.height()||
                    dx - wndSize / 2 < 0 || dx + wndSize / 2 >= lab2.width() ||
                    dy - wndSize / 2 < 0 || dy + wndSize / 2 >= lab2.height()) {
                return numeric_limits<float>::infinity();
            }

            /*
            int minSX = max(0, sx - wndSize / 2);
            int maxSX = min(lab1.width() - 1, sx + wndSize / 2);

            int minSY = max(0, sy - wndSize / 2);
            int maxSY = min(lab1.height() - 1, sy + wndSize / 2);

            int minDX = max(0, dx - wndSize / 2);
            int maxDX = min(lab2.width() - 1, dx + wndSize / 2);

            int minDY = max(0, dy - wndSize / 2);
            int maxDY = min(lab2.height() - 1, dy + wndSize / 2);

            // The extent of the valid window around (sx, sy) and (dx, dy)
            // to compare.
            int minX = -min(sx - minSX, dx - minDX);
            int maxX = min(maxSX - sx, maxDX - dx);

            int minY = -min(sy - minSY, dy - minDY);
            int maxY = min(maxSY - sy, maxDY - dy);
            */
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
                        // L1 norm:
                        lab1Diff += lDiff;
                        // L2 norm:
                        // lab1Diff = lDiff * lDiff;
                    }

                    float weight = exp(-(lab1Diff) / colorSigma);

                    cimg_forZC(lab1, z, c) {
                        float diff =
                            lab1(x + sx, y + sy, z, c) -
                            lab2(x + dx, y + dy, z, c);

                        diff = min(diff, maxDist);

                        totalWeight += weight;

                        // TODO Original paper also used a linear combination
                        //      of this and the difference in gradient.
                        ssd += diff * diff * weight;
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
function<bool(int, int, int, float[])> translationalCandidateGenerator(
    const CImg<float>& fieldLeft,
    const CImg<float>& fieldRight,
    int& iterationCounter,
    float randomSearchFactor = 1.0f,
    int increment = 1) {
    assert(fieldLeft.depth() == fieldRight.depth());

    int K = fieldLeft.depth();

    return [=](int x, int y, int i, float* value) -> bool {
        if (i > 3 * K) {
            return false;
        }

        if (i < K) {
            // Random sample
            float searchWndRadiusFactor = randomSearchFactor / pow(2.0f, iterationCounter);

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
                if (iterationCounter % 2 == 0) {
                    newX += increment;
                } else {
                    newX -= increment;
                }
            } else {
                z = i - 2 * K;
                // propagate up/down
                if (iterationCounter % 2 == 0) {
                    newY += increment;
                } else {
                    newY -= increment;
                }
            }

            if (!(newX < 0 || newX > fieldLeft.width() ||
                    newY < 0 || newY > fieldLeft.height())) {
                float newDisp = fieldLeft(newX, newY, z, 0);
                value[0] = newDisp;
            } else {
                value[0] = std::numeric_limits<float>::max();
            }
        } else if (i < 3 * K) {
            int newX = x, newY = y;

            int z = i - 2 * K;

            newX += fieldLeft(x, y, z, 0);

            if (!(newX < 0 || newX > fieldRight.width() ||
                    newY < 0 || newY > fieldRight.height())) {
                // TODO try different z values here too?
                int z2 = 0;

                value[0] = -fieldRight(newX, newY, z2, 0);
            } else {
                value[0] = std::numeric_limits<float>::max();
            }
        }

        return true;
    };
}

void patchMatchBPTranslationalCorrespondence(
        const CImg<float>& labLeft,
        const CImg<float>& labRight,
        CImg<float>& fieldLeft,
        CImg<float>& fieldRight,
        CImg<int>& fieldLeftSorted,
        CImg<int>& fieldRightSorted,
        CImg<float>& distLeft,
        CImg<float>& distRight,
        CImg<float>& unaryLeft,
        CImg<float>& unaryRight,
        CImg<float>& msgLeft,
        CImg<float>& msgRight,
        int wndSize = 15,
        int iterations = 5,
        float randomSearchFactor = 1.0f,
        int increment = 1,
        bool purePM = false) {
    // Counter to store the current iteration
    // Sampling functions will refer to this
    int iteration = 0;

    // Create functions to perform propagation and random sample
    auto sampleLeft = translationalCandidateGenerator(fieldLeft, fieldRight,
        iteration, randomSearchFactor, increment = 1);
    auto sampleRight = translationalCandidateGenerator(fieldRight, fieldLeft,
        iteration, randomSearchFactor, increment = 1);

    auto unaryCostLeft = translationalPatchDist(labLeft, labRight, wndSize);
    auto unaryCostRight = translationalPatchDist(labRight, labLeft, wndSize);

    auto binaryCost =
        [=](int aX, int aY, float* aVal, int bX, int bY, float* bVal) {
            return 0.0f;
        };

    for(; iteration < iterations; iteration++) {
        cout << "PMBP Iteration: " << iteration << endl;

        cout << "Left... ";
        patchMatchBeliefPropagation(
                fieldLeft, distLeft, fieldLeftSorted, unaryLeft,
                msgLeft, sampleLeft, unaryCostLeft, binaryCost,
                increment, purePM);
        cout << "done" << endl;

        cout << "Right... ";
        patchMatchBeliefPropagation(
                fieldRight, distRight, fieldRightSorted, unaryRight,
                msgRight, sampleRight, unaryCostRight, binaryCost,
                increment, purePM);
        cout << "done" << endl;
    }
}

/**
 * Implementation of extremely-generalized PatchMatch optimization of
 * arbitrary-dimension parameters over a 2D grid.
 *
 * Performs 2 iterations, propagating right and down on the first iteration,
 * and up and left on the second.
 *
 * Note that `field`, and `dist` must have the same width and height.
 *
 * `dist` must be 1 dimensional.
 *
 * `field` may have spectrum of arbitrary size, each channel representing
 * one element of the vector to optimize per pixel.  For example, a 2-channel
 * field may be used to optimize translational displacements.
 *
 * randomSample(int x, int y, float[] value) must randomly mutate the given
 * value.
 *
 * patchDist(int x, int y, float[] value) must return the error resulting
 * from assigning `value` to `field(x, y)`
 *
 * Returns the number of modified values.
 */
inline void patchMatch(
        CImg<float>& field,
        CImg<float>& dist,
        function<void(int, int, float[])> randomSample,
        function<float(int, int, float[])> patchDist,
        int increment = 1) {
    int propDirection = increment;

    for (int iter = 0; iter < 2; iter ++) {
        // Switch propagation direction during each iteration.
        propDirection *= -1;

#pragma omp parallel for
        for (int y = 0; y < field.height(); y += increment) {
            for (int x = 0; x < field.width(); x += increment) {
                // propagation
                // try the adjacent pixels along propDirection
                int adjY = y + propDirection;
                int adjX = x + propDirection;

                float tmp[field.spectrum()];

                if (adjY >= 0 && adjY < field.height()) {
                    bool different = false;

                    cimg_forC(field, c) {
                        different |= field(x, adjY, 0, c) != field(x, y, 0, c);
                        tmp[c] = field(x, adjY, 0, c);
                    }

                    if (different) {
                        float d = patchDist(x, y, tmp);

                        if (d < dist(x, y)) {
                            dbgOut << "Hor_Prop" << endl;
                            dist(x, y) = d;
                            cimg_forC(field, c) {
                                field(x, y, 0, c) = tmp[c];
                            }
                        }
                    }
                }

                if (adjX >= 0 && adjX < field.width()) {
                    bool different = false;

                    cimg_forC(field, c) {
                        different |= field(adjX, y, 0, c) != field(x, y, 0, c);
                        tmp[c] = field(adjX, y, 0, c);
                    }

                    if (different) {
                        float d = patchDist(x, y, tmp);

                        if (d < dist(x, y)) {
                            dbgOut << "Vert_Prop" << endl;
                            dist(x, y) = d;
                            cimg_forC(field, c) {
                                field(x, y, 0, c) = tmp[c];
                            }
                        }
                    }
                }

                
                // Random search
                cimg_forC(field, c) {
                    tmp[c] = field(x, y, 0, c);
                }

                randomSample(x, y, tmp);

                float d = patchDist(x, y, tmp);

                if (d < dist(x, y)) {
                    dbgOut << "RANDOM" << endl;
                    dist(x, y) = d;
                    cimg_forC(field, c) {
                        field(x, y, 0, c) = tmp[c];
                    }
                }
            }
        }
    }
}

/**
 * Extends patchMatch() to simultaneously solve for forward and reverse
 * mappings while performing left-right propagation.
 *
 * As a result, two new functions are necessary for converting offsets
 * and field values from the left to right, and right to left images.
 */
inline void patchMatchLeftRight(
        CImg<float>& fieldLeft,
        CImg<float>& fieldRight,
        CImg<float>& distLeft,
        CImg<float>& distRight,
        CImg<bool>& holesLeft,
        CImg<bool>& holesRight,
        function<void(int, int, float[])> randomSampleLeft,
        function<void(int, int, float[])> randomSampleRight,
        function<float(int, int, float[])> patchDistLeft,
        function<float(int, int, float[])> patchDistRight,
        function<void(int, int, float[], int&, int&, float[])> leftToRight,
        function<void(int, int, float[], int&, int&, float[])> rightToLeft,
        function<bool(int, int, float[], float[])> viewConsistent,
        int increment = 1) {
    // Perform PatchMatch on left and right sides
    patchMatch(fieldLeft, distLeft,
            randomSampleLeft, patchDistLeft, increment);
    patchMatch(fieldRight, distRight,
            randomSampleRight, patchDistRight, increment); 

    // Propagate from left to right
#pragma omp parallel for
    for (int y = 0; y < fieldLeft.height(); y += increment) {
        for (int x = 0; x < fieldLeft.width(); x += increment) {
            float lvalue[fieldLeft.spectrum()];

            cimg_forC(fieldLeft, c) {
                lvalue[c] = fieldLeft(x, y, 0, c);
            }

            int rx, ry;
            float rvalue[fieldRight.spectrum()];

            leftToRight(x, y, lvalue, rx, ry, rvalue);

            // Only modify pixels on the grid specified by increment
            rx += increment / 2;
            rx -= rx % increment;
            ry += increment / 2;
            ry -= ry % increment;

            holesLeft(x, y) = true;

            if (rx >= 0 && rx < fieldRight.width() &&
                    ry >= 0 && ry < fieldRight.height()) {
                holesRight(rx, ry) = true;

                float rvalueOld[fieldRight.spectrum()];

                cimg_forC(fieldRight, c) {
                    rvalueOld[c] = fieldRight(rx, ry, c);
                }

                if (viewConsistent(rx, ry, rvalue, rvalueOld)) {
                    holesLeft(x, y) = false;
                    holesRight(rx, ry) = false;
                } else {
                    float d = patchDistRight(rx, ry, rvalue);

                    if (d < distRight(x, y)) {
                        dbgOut << "LtR_View_Prop" << endl;
                        distRight(x, y) = d;

                        cimg_forC(fieldRight, c) {
                            fieldRight(x, y, 0, c) = rvalue[c];
                        }

                        holesLeft(x, y) = false;
                        holesRight(rx, ry) = false;
                    }
                }
            }
        }
    }
    
    // Propagate from right to left
#pragma omp parallel for
    for (int y = 0; y < fieldRight.height(); y += increment) {
        for (int x = 0; x < fieldRight.width(); x += increment) {
            float rvalue[fieldRight.spectrum()];

            cimg_forC(fieldRight, c) {
                rvalue[c] = fieldRight(x, y, 0, c);
            }
            
            int lx, ly;
            float lvalue[fieldLeft.spectrum()];

            rightToLeft(x, y, rvalue, lx, ly, lvalue);

            // Only modify pixels on the grid specified by increment
            lx += increment / 2;
            lx -= lx % increment;
            ly += increment / 2;
            ly -= ly % increment;

            holesRight(x, y) = true;

            if (lx >= 0 && lx < fieldLeft.width() &&
                    ly >= 0 && ly < fieldLeft.height()) {
                holesLeft(lx, ly) = true;

                float lvalueOld[fieldRight.spectrum()];

                cimg_forC(fieldLeft, c) {
                    lvalueOld[c] = fieldLeft(lx, ly, 0, c);
                }

                if (viewConsistent(lx, ly, lvalue, lvalueOld)) {
                    holesLeft(lx, ly) = false;
                    holesRight(x, y) = false;
                } else {
                    float d = patchDistLeft(lx, ly, lvalue);

                    if (d < distLeft(x, y)) {
                        dbgOut << "RtL_View_Prop" << endl;
                        distLeft(x, y) = d;

                        cimg_forC(fieldLeft, c) {
                            fieldLeft(x, y, 0, c) = lvalue[c];
                        }

                        holesLeft(lx, ly) = false;
                        holesRight(x, y) = false;
                    }
                }
            }
        }
    }
}

inline void fillHoles(
        CImg<float>& field,
        const CImg<bool>& holes,
        function<bool(float[], float[])> isBehind) {
#pragma omp parallel for
    for (int y = 0; y < field.height(); y++) {
        // Fill spans of holes by looking at the first non-hole on
        // the left and right, and choosing the lesser of the two
        int holeSpanStart = -1;
        for (int x = 0; x <= field.width(); x++) {
            if (x < field.width() && holes(x, y)) {
                if (holeSpanStart == -1) {
                    holeSpanStart = x;
                }
            } else {
                if (holeSpanStart != -1) {
                    float leftNonHole[field.spectrum()];
                    float rightNonHole[field.spectrum()];

                    float* holeFiller = NULL;

                    if (holeSpanStart > 0) {
                        cimg_forC(field, c) {
                            leftNonHole[c] = field(holeSpanStart - 1, y, 0, c);
                        }
                    }
    
                    if (x < field.width()) {
                        cimg_forC(field, c) {
                            rightNonHole[c] = field(x, y, 0, c);
                        }
                    }

                    if (holeSpanStart > 0 && x >= field.width()) {
                        holeFiller = leftNonHole;
                    } else if (holeSpanStart <= 0 && x < field.width()) {
                        holeFiller = rightNonHole;
                    } else if (holeSpanStart > 0 && x < field.width()) {
                        holeFiller = isBehind(leftNonHole, rightNonHole)?
                            leftNonHole : rightNonHole;
                    }

                    if (holeFiller != NULL) {
                        for (int i = holeSpanStart; i < x; i++) {
                            cimg_forC(field, c) {
                                field(i, y, 0, c) = holeFiller[c];
                            }
                        }
                    }

                    holeSpanStart = -1;
                }
            }
        }
    }
}

/**
 * Uses PatchMatch to solve for translational correspondence (without slanted
 * support windows) with integer disparity precision.
 */
inline void patchMatchTranslationalCorrespondence(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        CImg<float>& fieldLeft,
        CImg<float>& fieldRight,
        CImg<float>& distLeft,
        CImg<float>& distRight,
        CImg<bool>& holesLeft,
        CImg<bool>& holesRight,
        int wndSize = 15,
        int iterations = 5,
        float randomSearchFactor = 1.0f,
        int increment = 1) {
    int iter = 0;

    function<void(int, int, float*)> sampleLeft, sampleRight;
    // Repeat to generate sampleLeft and sampleRight
    for (int i = 0; i < 2; i++) {
        int width = (i == 0) ? lab2.width() : lab1.width();

        auto sample = [&lab1, &lab2, &iter, width, wndSize, randomSearchFactor]
            (int sx, int sy, float* value) {
                float searchWndRadiusFactor = randomSearchFactor / pow(2.0f, iter);

                float searchWndWidth  = searchWndRadiusFactor * width;

                float minSearchWndX = sx + value[0] - searchWndWidth / 2.0f;

                float maxSearchWndX = sx + value[0] + searchWndWidth / 2.0f;

                minSearchWndX = max(0.0f, minSearchWndX);

                maxSearchWndX = max((float) width, maxSearchWndX);

                // The point we have chosen to randomly sample from
                int randX = (int) (cimg::rand() * (maxSearchWndX - minSearchWndX) + minSearchWndX);

                value[0] = randX - sx;
            };

        if (i == 0) {
            sampleLeft = sample;
        } else {
            sampleRight = sample;
        }
    }

    auto patchDistLeft = translationalPatchDist(lab1, lab2, wndSize);
    auto patchDistRight = translationalPatchDist(lab2, lab1, wndSize);

    auto flip = []
        (int rx, int ry, float* rvalue,
         int& lx, int& ly, float* lvalue) {
            lx = rx + rvalue[0];
            ly = ry;
            lvalue[0] = -rvalue[0];
    };

    auto viewConsistent = [increment]
        (int x, int y, float* value1, float* value2) -> bool {
            return abs(value1[0] - value2[0]) <= increment * 2;
    };

    for (; iter < iterations; iter++) {
        printf("Processing iteration: %d\n", iter);
        patchMatchLeftRight(
                fieldLeft, fieldRight,
                distLeft, distRight,
                holesLeft, holesRight,
                sampleLeft, sampleRight,
                patchDistLeft, patchDistRight,
                flip, flip,
                viewConsistent,
                increment);
    }
}

inline function<float(int, int, float[])> affinePatchDist(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        int wndSize,
        float colorSigma = 30.0f,
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
                            lab1Diff += lDiff * lDiff;
                        }
                        float weight = exp(-(lab1Diff) / colorSigma);

                        if (dstX >= 0 && dstX < lab2.width() &&
                                dstY >= 0 && dstY < lab2.height()) {
                            cimg_forZC(lab1, z, c) {
                                float diff = lab1(srcX, srcY, z, c) -
                                    lab2.linear_atXYZC(dstX, dstY, z, c);

                                diff = min(diff, maxDist);

                                ssd += diff * diff * weight;
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


/**
 * Same as patchMatchTranslationalCorrespondence, but solves for
 * an affine transform where field contains 3 values (a, b, c)
 * and the effective translation for each point (srcX, srcY) -> (dstX, dstY) is
 *
 * (dstX, dstY) = (srcX + a + b * srcX + c * srcY, srcY)
 *
 * This also solves for sub-pixel precision.
 */
inline void patchMatchAffine(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        CImg<float>& fieldLeft,
        CImg<float>& fieldRight,
        CImg<float>& distLeft,
        CImg<float>& distRight,
        CImg<bool>& holesLeft,
        CImg<bool>& holesRight,
        int wndSize = 15,
        int iterations = 5,
        float randomSearchFactor = 1.0f,
        float maxAngleDeviation = 9.0f,
        int increment = 1) {
    int iter = 0;

    int width = lab2.width();
    auto sampleLeft = 
        [&iter, width, wndSize, randomSearchFactor, maxAngleDeviation]
        (int sx, int sy, float* value) {
            float oldDstX = sx + value[0] + sx * value[1] + sy * value[2];

            float searchWndRadiusFactor = randomSearchFactor / (iter + 1);

            float searchWndWidth  = searchWndRadiusFactor * width;

            float minNewDstX = (oldDstX - searchWndWidth / 2.0f);
            minNewDstX = max(0.0f, minNewDstX);

            float maxNewDstX = (oldDstX + searchWndWidth / 2.0f);
            maxNewDstX = min((float) width, maxNewDstX);

            float newDstX = cimg::rand() * (maxNewDstX - minNewDstX) +
                minNewDstX;

            float b, c;

            // This is equivalent to sampling a random normal vector
            // and computing nx/nz and ny/nz as per the original paper.
            float rand = (cimg::rand()*2.0f - 1.0f) * maxAngleDeviation;
            b = tan(rand * cimg::PI / 2.0f);
            rand = (cimg::rand()*2.0f - 1.0f) * maxAngleDeviation;
            c = tan(rand * cimg::PI / 2.0f);

            value[0] = newDstX - (sx + b * sx + c * sy);
            value[1] = b;
            value[2] = c;
        };

    width = lab1.width();
    auto sampleRight =
        [&iter, width, wndSize, randomSearchFactor, maxAngleDeviation]
        (int sx, int sy, float* value) {
            float oldDstX = sx + value[0] + sx * value[1] + sy * value[2];

            float searchWndRadiusFactor = randomSearchFactor / (iter + 1);

            float searchWndWidth  = searchWndRadiusFactor * width;

            float minNewDstX = (oldDstX - searchWndWidth / 2.0f);
            minNewDstX = max(0.0f, minNewDstX);

            float maxNewDstX = (oldDstX + searchWndWidth / 2.0f);
            maxNewDstX = min((float) width, maxNewDstX);

            float newDstX = cimg::rand() * (maxNewDstX - minNewDstX) +
                minNewDstX;

            float b, c;

            // This is equivalent to sampling a random normal vector
            // and computing nx/nz and ny/nz as per the original paper.
            float rand = (cimg::rand()*2.0f - 1.0f) * maxAngleDeviation;
            b = tan(rand * cimg::PI / 2.0f);
            rand = (cimg::rand()*2.0f - 1.0f) * maxAngleDeviation;
            c = tan(rand * cimg::PI / 2.0f);

            value[0] = newDstX - (sx + b * sx + c * sy);
            value[1] = b;
            value[2] = c;
        };

    auto patchDistLeft = affinePatchDist(lab1, lab2, wndSize);
    auto patchDistRight = affinePatchDist(lab2, lab1, wndSize);

    auto flip = []
        (int rx, int ry, float* rvalue,
         int& lx, int& ly, float* lvalue) {
            // FIXME this is wrong
            lx = rx + rvalue[0] + rx * rvalue[1] + ry * rvalue[2];
            ly = ry;
            float b = -rvalue[1];
            float c = rvalue[2];
            lvalue[0] = rx - (lx + b * lx + c * ly);
            lvalue[1] = b;
            lvalue[2] = c;
    };

    auto viewConsistent = [increment]
        (int x, int y, float* value1, float* value2) -> bool {
            float x1 = x + value1[0] + value1[1] * x + value1[2] * y;
            float x2 = x + value2[0] + value2[1] * x + value2[2] * y;
            return abs(x1 - x2) <= increment * 2;
    };

    for (; iter < iterations; iter++) {
        printf("Processing iteration: %d\n", iter);
        patchMatchLeftRight(
                fieldLeft,
                fieldRight,
                distLeft,
                distRight,
                holesLeft,
                holesRight,
                sampleLeft,
                sampleRight,
                patchDistLeft,
                patchDistRight,
                flip, flip,
                viewConsistent,
                increment);
    }
}

int main(int argc, char** argv) {
    int sampleIndex = 0;
    if (argc >= 2) {
        sampleIndex = atoi(argv[1]);
    }
    for (; sampleIndex < NUM_SAMPLES; sampleIndex++) {
        CImg<float> fst(SAMPLES[sampleIndex][0].c_str());
        CImg<float> lst(SAMPLES[sampleIndex][1].c_str());

        int MAX_SIZE = 1024;

        while (fst.width() > MAX_SIZE || fst.height() > MAX_SIZE) {
            fst.resize_halfXY();
            lst.resize_halfXY();
        }

        // The number of "particles" to use
        int K = 2;

        CImg<float> labLeft = fst.get_RGBtoLab();
        CImg<float> labRight  = lst.get_RGBtoLab();

        // TEST for translationalPatchDist
        if (false) {
            CImg<float> dist(labLeft.width(), labLeft.height(), 1);
            auto metric = translationalPatchDist(labLeft, labRight, 15);
            float value = 0;
            cimg_forXY(dist, x, y) {
                dist(x, y) = metric(x, y, &value);
            }
            dist.display();
            return 0;
        }

        CImg<float> fieldLeft(labLeft.width(), labLeft.height(), K, 1);
        CImg<float> fieldRight(labRight.width(), labRight.height(), K, 1);
        // Initialize with random, (mostly) valid disparity particles
        cimg_forXYZ(fieldLeft, x, y, z) {
            int randX = (int) (cimg::rand() * labRight.width());

            fieldLeft(x, y, z) = randX - x;
        }
        cimg_forXYZ(fieldRight, x, y, z) {
            int randX = (int) (cimg::rand() * labLeft.width());

            fieldLeft(x, y, z) = randX - x;
        }

        CImg<int> fieldLeftSorted(labLeft.width(), labLeft.height(), K);
        CImg<int> fieldRightSorted(labRight.width(), labRight.height(), K);
        cimg_forXYZ(fieldLeftSorted, x, y, z) {
            fieldLeftSorted(x, y, z) = z;
        }
        cimg_forXYZ(fieldRightSorted, x, y, z) {
            fieldRightSorted(x, y, z) = z;
        }

        CImg<float> distLeft(labLeft.width(), labLeft.height(), K);
        CImg<float> distRight(labRight.width(), labRight.height(), K);

        distLeft = std::numeric_limits<float>::max();
        distRight = std::numeric_limits<float>::max();

        CImg<float> unaryLeft(labLeft.width(), labLeft.height(), K);
        CImg<float> unaryRight(labRight.width(), labRight.height(), K);

        unaryLeft = std::numeric_limits<float>::max();
        unaryRight = std::numeric_limits<float>::max();

        CImg<float> msgLeft(labLeft.width(), labLeft.height(), K, 4);
        msgLeft = 0.0f;
        CImg<float> msgRight(labRight.width(), labRight.height(), K, 4);
        msgRight = 0.0f;

        int wndSize = 7;
        int iterations = 3;
        float randomSearchFactor = 1.0f;
        int increment = 1;
        patchMatchBPTranslationalCorrespondence(
                labLeft,         labRight,
                fieldLeft,       fieldRight,
                fieldLeftSorted, fieldRightSorted,
                distLeft,        distRight,
                unaryLeft,       unaryRight,
                msgLeft,         msgRight,
                wndSize,
                iterations,
                randomSearchFactor,
                increment,
                true);

        // Display the result, after filering for particles with a particular
        // rank.
        CImg<float> fieldLeftSlice(fieldLeft.width(), fieldLeft.height());
        CImg<float> fieldRightSlice(fieldRight.width(), fieldRight.height());

        for (int i = 0; i < K; i++) {
            cout << "Particles with rank " << i << endl;
            cimg_forXY(fieldLeftSlice, x, y) {
                fieldLeftSlice(x, y) =
                    fieldLeft(x, y, fieldLeftSorted(i));
            }
            cimg_forXY(fieldRightSlice, x, y) {
                fieldRightSlice(x, y) =
                    fieldRight(x, y, fieldRightSorted(i));
            }
            fieldLeftSorted.get_shared_slice(0).display();
            fieldLeftSorted.get_shared_slice(1).display();
            fieldRightSorted.get_shared_slice(0).display();
            fieldRightSorted.get_shared_slice(1).display();

            CImgList<float>(
                    fieldLeftSlice
                        .get_normalize(0.0f, 255.0f)
                        .get_equalize(256)
                        .get_map(CImg<float>().jet_LUT256()),
                    fieldRightSlice
                        .get_normalize(0.0f, 255.0f)
                        .get_equalize(256)
                        .get_map(CImg<float>().jet_LUT256())
                    ).display();
        }

        /*
        CImg<float> fieldLeft(fst.width(), fst.height(), 1, 3);
        CImg<float> fieldRight(lst.width(), lst.height(), 1, 3);
        CImg<float> errorLeft(fst.width(), fst.height());
        CImg<float> errorRight(lst.width(), lst.height());
        CImg<bool> holesLeft(fst.width(), fst.height());
        CImg<bool> holesRight(lst.width(), lst.height());

        fieldLeft = 0.0f;
        fieldRight = 0.0f;
        errorLeft = std::numeric_limits<float>::infinity();
        errorRight = std::numeric_limits<float>::infinity();

        CImg<float> lab1 = fst.get_RGBtoLab();
        CImg<float> lab2 = lst.get_RGBtoLab();

        CImg<float> fieldLeftTran = fieldLeft.get_shared_channel(0);
        CImg<float> fieldRightTran = fieldRight.get_shared_channel(0);

        int wndSize = 21;
        float searchFact = 1.0f;
        for (int increment = 16; increment >= 1; increment /= 2) {
            printf("Running Translational Correspondence with "
                    "inc = %d, searchFact = %f\n",
                    increment, searchFact);
            patchMatchTranslationalCorrespondence(
                    lab1, lab2,
                    fieldLeftTran, fieldRightTran,
                    errorLeft, errorRight,
                    holesLeft, holesRight,
                    wndSize, 2, searchFact, increment);
            searchFact /= 2.0f;
        }

        fieldLeft.get_shared_channel(0).get_normalize(0.0f, 255.0f).get_equalize(256).get_map(CImg<float>().jet_LUT256()).display();

        fieldLeft.get_shared_channel(1) = 0.0f;
        fieldLeft.get_shared_channel(2) = 0.0f;
        fieldRight.get_shared_channel(1) = 0.0f;
        fieldRight.get_shared_channel(2) = 0.0f;

        patchMatchAffine(
                lab1, lab2,
                fieldLeft, fieldRight,
                errorLeft, errorRight,
                holesLeft, holesRight,
                7, 2, 0.10f, 0.0f, 1);

        patchMatchAffine(
                lab1, lab2,
                fieldLeft, fieldRight,
                errorLeft, errorRight,
                holesLeft, holesRight,
                5, 2, 0.05f, 0.0f, 1);
        */

        /*
        patchMatchAffine(
                lab1, lab2,
                fieldLeft, fieldRight,
                errorLeft, errorRight,
                holesLeft, holesRight,
                21, 1, 0.0f, 0.5f);
        */

        /*
        // Fill holes with 0 for visualization purposes
        cimg_forXY(fieldLeft, x, y) {
            if (holesLeft(x, y)) {
                cimg_forZC(fieldLeft, z, c) {
                    fieldLeft(x, y, z, c) = 0.0f;
                }
            }
        }
        cimg_forXY(fieldRight, x, y) {
            if (holesRight(x, y)) {
                cimg_forZC(fieldRight, z, c) {
                    fieldRight(x, y, z, c) = 0.0f;
                }
            }
        }

        // Convert from affine slanted-support values to scalar disparities
        CImg<float> dispLeft(fieldLeft.width(), fieldLeft.height());
        cimg_forXY(dispLeft, x, y) {
            dispLeft(x, y, 0) = fieldLeft(x, y, 0, 0) + fieldLeft(x, y, 0, 1) * x + fieldLeft(x, y, 0, 2) * y;
        }

        CImg<float> dispRight(fieldRight.width(), fieldRight.height());
        cimg_forXY(dispRight, x, y) {
            dispRight(x, y, 0) = fieldRight(x, y, 0, 0) + fieldRight(x, y, 0, 1) * x + fieldRight(x, y, 0, 2) * y;
        }

        // Display the result
        CImgDisplay hlDisp(holesLeft);
        CImgDisplay hrDisp(holesRight);
        CImgList<float>(
                dispLeft,
                dispRight
            ).display();

        */

        /*
        CImgDisplay fstDisp(fst);
        CImgDisplay lstDisp(lst);

        // Display a histogram of disparities 
        // float fieldMin, fieldMax;
        // fieldMin = fieldLeft.min_max(fieldMax);
        // printf("\n\n\nmin = %f, max = %f\n\n\n", fieldMin, fieldMax);
        // fieldLeft.get_histogram(fieldMax - fieldMin + 2, fieldMin - 1, fieldMax + 1).display_graph(0, 3);

        fieldLeft.blur_median(3);
        fieldRight.blur_median(3);

        CImgDisplay leftHolesDisp(holesLeft);
        CImgDisplay rightHolesDisp(holesRight);

        CImgList<float>(
                fieldRight.get_equalize(256),
                fieldLeft.get_equalize(256)).display();
        
        visualizeCorrespondence(fst, fieldLeft, lst);
        */


        // visualizeCorrespondence(fst, disp, lst);

        /*
        CImg<float> matches;
        featureMatchCV(
                fst.get_RGBtoLab().channel(0),
                lst.get_RGBtoLab().channel(0),
                matches);

        matches.display();
        */
        // CImgDisplay lstDisp(lst);
        // CImgDisplay errDisp(error);
    }

    return 1;
}


/**
 * Computes optical flow from a to b and places the displacement field
 * into flow with x and y components in depth planes 0 and 1, respectively.
 */
void flowLK(
        const CImg<float>& a,
        const CImg<float>& b,
        float sigma,
        CImg<float>& flow) {
    assert(a.width() == b.width() && a.width() == flow.width() &&
            a.height() == b.height() && a.height() == flow.height());

    assert(flow.depth() == 2);

    // This uses the least-squares method shown in
    // http://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method

    CImgList<float> spaceGrad = a.get_gradient();
    const CImg<float>& Ix = spaceGrad(0);
    const CImg<float>& Iy = spaceGrad(1);

    CImg<float> It(a.width(), a.height());
    It = b - a;

    CImg<float> A00 = Ix.get_sqr();
    CImg<float> Adiag = Ix.get_mul(Iy);
    CImg<float> A11 = Iy.get_sqr();

    CImg<float> B0 = Ix.get_mul(It) * -1;
    CImg<float> B1 = Iy.get_mul(It) * -1;

    // Apply weighting function, this is equivalent to blurring for which
    // CImg has a faster-than-naive-gaussian implementation.
    A00.blur(sigma);
    Adiag.blur(sigma);
    A11.blur(sigma);

    // Compute results by inverting A and multiplying with B
    cimg_forXY(flow, x, y) {
        float det = A00(x, y) * A11(x, y) - Adiag(x, y) * Adiag(x, y);
        float AInv00   =  A11(x, y)   / det;
        float AInvdiag = -Adiag(x, y) / det;
        float AInv11   =  A00(x, y)   / det;

        flow(x, y, 0) = AInv00   * B0(x, y) + AInvdiag * B1(x, y);
        flow(x, y, 1) = AInvdiag * B1(x, y) + AInv11   * B1(x, y);
    }
}

void stereoBMCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& flow) {
    assert(a.depth() == 1 && a.spectrum() == 1 &&
            b.depth() == 1 && b.spectrum() == 1);

    assert(a.width() == b.width() && a.width() == flow.width() &&
            a.height() == b.height() && a.height() == flow.height());

    assert(flow.depth() == 2);

    // Normalize before passing to OpenCV
    CImg<float> aNorm = a.get_normalize(0.0f, 1.0f);
    CImg<float> bNorm = b.get_normalize(0.0f, 1.0f);

    // cv::GpuMat cvA(aNorm.height(), aNorm.width(), CV_32FC1, (void*) aNorm.data());

    // cv::GpuMat cvB(bNorm.height(), bNorm.width(), CV_32FC1, (void*) bNorm.data());

    // cv::GpuMat cvFlow(flow.height(), flow.width(), CV_32FC2, (void*) flow.data);

    // cv::StereoConstantSpaceBP bp;

    // bp(cvA, cvB, cvFlow);

}

void flowCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& flow) {
    assert(a.depth() == 1 && a.spectrum() == 1 &&
            b.depth() == 1 && b.spectrum() == 1);

    assert(a.width() == b.width() && a.width() == flow.width() &&
            a.height() == b.height() && a.height() == flow.height());

    assert(flow.depth() == 2);

    // Normalize before passing to OpenCV
    CImg<float> aNorm = a.get_normalize(0.0f, 1.0f);
    CImg<float> bNorm = b.get_normalize(0.0f, 1.0f);

    cv::Mat cvA(aNorm.height(), aNorm.width(), CV_32FC1, (void*) aNorm.data());

    cv::Mat cvB(bNorm.height(), bNorm.width(), CV_32FC1, (void*) bNorm.data());

    cv::Mat cvFlow(flow.height(), flow.width(), CV_32FC2);

    double pyr_scale = 0.5;
    int levels = 10;
    int winsize = 3;
    int iterations = 5;
    int poly_n = 5;
    double poly_sigma = 1.5;
    int flags = 0;
    cv::calcOpticalFlowFarneback(cvA, cvB, cvFlow, 
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

    cv::Mat cvFlowX(flow.height(), flow.width(), CV_32FC1,
            (void*) flow.data());
    cv::Mat cvFlowY(flow.height(), flow.width(), CV_32FC1,
            (void*) (flow.data() + flow.width() * flow.height()));

    std::vector<cv::Mat> output;
    output.push_back(cvFlowX);
    output.push_back(cvFlowY);

    cv::split(cvFlow, output);
}

/**
 * Uses OpenCV to match interest points in Grayscale images 'a' and 'b'.
 *
 * The result is returned in matches as a set of column vectors containing
 * (a.x, a.y, b.x, b.y).
 *
 * See http://docs.opencv.org/doc/tutorials/features2d/feature_description/feature_description.html#feature-description
 */
void featureMatchCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& matches) {
    assert(a.depth() == 1 && a.spectrum() == 1 &&
            b.depth() == 1 && b.spectrum() == 1);

    CImg<float> aNorm = a.get_normalize(0.0f, 255.0f);
    CImg<float> bNorm = b.get_normalize(0.0f, 255.0f);

    cv::Mat cvA(aNorm.height(), aNorm.width(), CV_32FC1, (void*) aNorm.data());

    cv::Mat cvB(bNorm.height(), bNorm.width(), CV_32FC1, (void*) bNorm.data());

    cv::Mat cvA8;
    cvA.convertTo(cvA8, CV_8U);

    cv::Mat cvB8;
    cvB.convertTo(cvB8, CV_8U);

    cv::ORB orb;

    std::vector<cv::KeyPoint> keypointsA, keypointsB;
    cv::Mat descriptorsA, descriptorsB;

    orb(cvA8, cv::Mat(), keypointsA, descriptorsA);
    orb(cvB8, cv::Mat(), keypointsB, descriptorsB);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matchList;
    matcher.match(descriptorsA, descriptorsB, matchList);

    matches = CImg<float>(matchList.size(), 4);

    cimg_forX(matches, i) {
        matches(i, 0) = keypointsA[matchList[i].queryIdx].pt.x;
        matches(i, 1) = keypointsA[matchList[i].queryIdx].pt.y;
        matches(i, 2) = keypointsB[matchList[i].trainIdx].pt.x;
        matches(i, 3) = keypointsB[matchList[i].trainIdx].pt.y;
    }
}
