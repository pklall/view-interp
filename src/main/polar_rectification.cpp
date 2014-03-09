#include "polar_rectification.h"

PolarRectification::PolarRectification(
        int _width,
        int _height,
        Eigen::Matrix3f _F,
        array<Eigen::Vector2f, 2> _match) :
    imgWidth(_width),
    imgHeight(_height),
    F(_F),
    match(_match) {

    epipoles[0] = F.fullPivLu().kernel().col(0).hnormalized();
    epipoles[1] = F.transpose().fullPivLu().kernel().col(0).hnormalized();
}

void PolarRectification::getRelevantEdges(
        int imgId,
        vector<Eigen::ParametrizedLine<float, 2>>& edges) const {
    const auto& epipole = epipoles[imgId];

    // All 4 lines are parameterized in counter-clockwise order: top, left,
    // right, bottom.
    array<Eigen::ParametrizedLine<float, 2>, 4> allEdges;

    // top edge facing left
    allEdges[0] = Eigen::ParametrizedLine<float, 2>(
            Eigen::Vector2f(imgWidth, 0),
            Eigen::Vector2f(-imgWidth, 0));

    // left edge facing down
    allEdges[1] = Eigen::ParametrizedLine<float, 2>(
            Eigen::Vector2f(0, 0),
            Eigen::Vector2f(0, imgHeight));

    // bottom edge facing right
    allEdges[2] = Eigen::ParametrizedLine<float, 2>(
            Eigen::Vector2f(0, imgHeight),
            Eigen::Vector2f(imgWidth, 0));
    
    // right edge facing up
    allEdges[3]  = Eigen::ParametrizedLine<float, 2>(
            Eigen::Vector2f(imgWidth, imgHeight),
            Eigen::Vector2f(0, -imgHeight));
    

    edges.clear();
    edges.reserve(4);

    // Determine which edges are relevant
    array<bool, 4> rel;

    // top
    rel[0] = epipole.y() > 0;

    // left
    rel[1] = epipole.x() > 0;

    // bottom
    rel[2] = epipole.y() < imgHeight;

    // right
    rel[3] = epipole.x() < imgWidth;

    // Push the relevant edges in head-to-tail order...
    
    // The first edge to insert must have a non-relevant edge before it.  This
    // ensures that when we insert edges into the final list, we have a
    // contiguous set of edges parameterized from head-to-tail without gaps.
    int startIndex = 0;

    for (int i = 0; i < 4; i++) {
        if (!rel[(i - 1 + 4) % 4] && rel[i]) {
            startIndex = i;
        }
    }

    for (int i = 0; i < 4; i++) {
        edges.push_back(allEdges[(startIndex + i) % 4]);
    }
}

void PolarRectification::getEpipolarLine(
        int imgId,
        const Eigen::Vector2f& originalPt,
        Eigen::Vector2f& line) const {
    // Flip the problem when necessary.
    const auto& e0 = (imgId == 0) ? epipoles[0] : epipoles[1];
    const auto& e1 = (imgId == 0) ? epipoles[1] : epipoles[0];

    Eigen::Matrix3f fundMat;

    if (imgId == 0) {
        fundMat = F;
    } else {
        fundMat = F.transpose();
    }

    const auto& match0 = (imgId == 0) ? match[0] : match[1];
    const auto& match1 = (imgId == 0) ? match[1] : match[0];

    Eigen::Vector2f lineOther;

    auto& l0 = (imgId == 0) ? line : lineOther;
    auto& l1 = (imgId == 0) ? lineOther : line;

    // Procede to compute lines as if we are given a point in image 0.

    l0 = originalPt - e0;

    Eigen::Vector3f line1DirH = fundMat * originalPt.homogeneous();
    l1 = line1DirH.hnormalized().unitOrthogonal();

    // Project the known match onto each line
    auto line0MatchProj = (match0 - e0).transpose() * l0;

    auto line1MatchProj = (match1 - e1).transpose() * l1;

    // Flip the direction of line 1 if necessary to select the correct
    // half-epipolar line.
    if (line0MatchProj > 0 != line1MatchProj > 0) {
        l1 *= -1;
    }
}

bool PolarRectification::getImg0ClippingPlanes(
        array<Eigen::Vector2f, 2>& planes) const {
    Eigen::AlignedBox<float, 2> img1(
            Eigen::Vector2f(0, 0), 
            Eigen::Vector2f(imgWidth, imgHeight));

    if (img1.contains(epipoles[1])) {
        return false;
    }

    array<Eigen::Vector2f, 4> eLines;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2f corner;

        // Enumerate all corners of the image in binary
        corner[0] = (i & 0x1) == 0 ? 0 : imgWidth;
        corner[1] = (i & 0x2) == 0 ? 0 : imgHeight;

        getEpipolarLine(1, corner, eLines[i]);
    }

    // The clipping planes are the only two planes such that all
    // other projected epipolar half-lines will be on the same side.
    // eLines specifies epipolar line directions.
    int curCandidate = 0;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2f candidatePlane = eLines[i].unitOrthogonal();

        int numNonNeg = 0;

        for (int j = 0; j < 4; j++) {
            if (j == i) {
                continue;
            }

            // Since these vectors can also be interpreted as points on the
            // epipolar lines after translation such that the epipole is the
            // origin, we can conveniently perform projection in this space.
            bool nonNeg = eLines[j].transpose() * candidatePlane >= 0;

            if (nonNeg) {
                numNonNeg++;
            }
        }

        if (numNonNeg == 3) {
            planes[curCandidate] = candidatePlane;

            curCandidate++;
        } else if (numNonNeg == 0) {
            planes[curCandidate] = -1 * candidatePlane;

            curCandidate++;
        }
        
        if (curCandidate == 2) {
            break;
        }
    }

    return true;
}

void PolarRectification::createRectificationMap(
        int maxPixelsPerLine,
        vector<Eigen::Vector2f>& endpoints) const {
    /*
     * To determine the set of epipolar lines to emmit, do the following:
     *  - Loop over all edges of image 0 which are furthest from the epipolar
     *    line (these are saved in variable `edges`) in order such that
     *  - If the epipole is outside in image 1, then clip the current
     *    edge's domain against the range of t-values associated with the
     *    intersections of the line with the epipolar lines associated with
     *    each corner point in image 1.
     *  - Walk along the edge, stepping maxPixelsPerLine, each time.
     *  - Determine the distance between the previous edge point and the
     *    new epipolar line.
     */

    vector<Eigen::ParametrizedLine<float, 2>> edges0;
    vector<Eigen::ParametrizedLine<float, 2>> edges1;

    getRelevantEdges(0, edges0);
    getRelevantEdges(1, edges1);

    array<Eigen::Vector2f, 2> img0Clip;

    bool mustClip = getImg0ClippingPlanes(img0Clip);

    for (const auto& edge : edges0) {
        float tmin = 0.0f;
        float tmax = 1.0f;

        if (mustClip) {
            // TODO adjust tmin and tmax
        }

        float curT = tmin;

        const float edgeLength = (edge.pointAt(1) - edge.pointAt(0)).norm();

        const float maxStep = 1.0f / edgeLength;

        while (curT < tmax) {
            const Eigen::Vector2f curPt = edge.pointAt(curT);

            float step = maxStep;

            // TODO
            // Project curPt's epipolar line onto image 1's relevant edges -> x1'
            // Project curPt + step epipolar line onto image 1's relevant edges -> x2'
            // Find maximum step in direction from x1' to x2' along image 1 edge
            // associated with x1' -> update x2' - also clamp against
            // endpoints of edge in image1.
            //
            // Back project x2' into image 0 and find associated T-value.
            //
            // step = min(step, t-value - curT)
            
            // TODO emit new endpoint

            curT += step;
        }
    }
}

