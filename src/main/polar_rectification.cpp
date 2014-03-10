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
            for (const Eigen::Vector2f& normal : img0Clip) {
                const Eigen::Vector2f edgeStart = edge.pointAt(tmin);
                const Eigen::Vector2f edgeEnd = edge.pointAt(tmax);

                Eigen::Hyperplane<float, 2> plane(normal, epipoles[0]);

                float intersectT = edge.intersectionParameter(plane);

                if (plane.signedDistance(edge.pointAt(0)) < 0) {
                    tmin = max(intersectT, tmin);
                }

                if (plane.signedDistance(edge.pointAt(1)) < 0) {
                    tmax = min(intersectT, tmax);
                }
            }
        }

        float curT = tmin;

        const float edgeLength = (edge.pointAt(1) - edge.pointAt(0)).norm();

        const float maxStep = 1.0f / edgeLength;

        while (curT < tmax) {
            const Eigen::Vector2f curPt = edge.pointAt(curT);

            float step = maxStep;

            const Eigen::Vector2f nextPt = edge.pointAt(curT + step);

            // curPt's epipolar line in image 1
            Eigen::Vector2f curPtL1;

            getEpipolarLine(0, curPt, curPtL1);

            Eigen::Hyperplane<float, 2> curPtL1Plane(
                    curPtL1.unitOrthogonal(),
                    epipoles[1]);

            // nextPt's epipolar line in image 1
            Eigen::Vector2f nextPtL1;

            getEpipolarLine(0, nextPt, nextPtL1);

            Eigen::Hyperplane<float, 2> nextPtL1Plane(
                    nextPtL1.unitOrthogonal(),
                    epipoles[1]);

            // Find the relevant edge in image 1 which intersects curPtL1
            Eigen::ParametrizedLine<float, 2>* img1Edge = &(edges1[0]);

            for (int i = 0; i < edges1.size(); i++) {
                float intersectionT = edges1[i].intersectionParameter(curPtL1Plane);

                if (intersectionT >= 0 && intersectionT <= 1) {
                    img1Edge = &(edges1[i]);
                }
            }

            // Find the points corresponding to of curPt and nextPt in image 1
            // along the edge of the image.
            Eigen::Vector2f curPt1 = img1Edge->intersectionPoint(curPtL1Plane);
            Eigen::Vector2f nextPt1 = img1Edge->intersectionPoint(nextPtL1Plane);

            // The point along the image 1 edge corresponding to the maximum
            // allowed step.
            Eigen::Vector2f maxStepPt1 = 
                (nextPt1 - curPt1).normalized() * maxPixelsPerLine + curPt1;

            // maxStepPt1's epipolar line in image 0
            Eigen::Vector2f maxStepPtL0;

            getEpipolarLine(1, maxStepPt1, maxStepPtL0);

            Eigen::Hyperplane<float, 2> maxStepPtL0Plane(
                    maxStepPtL0.unitOrthogonal(),
                    epipoles[0]);

            // The intersection of this epipolar line and the current edge
            float maxStepT = edge.intersectionParameter(maxStepPtL0Plane);

            step = min(maxStep, maxStepT - curT);

            assert(step > 0);

            curT += step;

            curT = min(curT, tmax);

            endpoints.push_back(edge.pointAt(curT));
        }
    }
}

void PolarRectification::getEpipolarDistanceRanges(
        int imgId,
        float& rmin,
        float& rmax) const {
    const Eigen::Vector2f e = epipoles[imgId];

    rmin = std::numeric_limits<float>::max();
    rmax = std::numeric_limits<float>::min();

    // Corners in counter-clockwise order
    Eigen::Matrix<float, 2, 4> corners;
    corners <<
        0, 0,         imgWidth,  imgWidth,
        0, imgHeight, imgHeight, 0;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2f corner0 = corners.col(i);
        Eigen::Vector2f corner1 = corners.col((i + 1) % 4);

        Eigen::ParametrizedLine<float, 2> edge(corner0, corner1 - corner0);

        Eigen::Hyperplane<float, 2> epipoleEdgePlane(
                (corner1 - corner0).normalized(), e);

        // This is the parameter associated with the closest point on the
        // edge to the epipole.
        float intersectT = edge.intersectionParameter(epipoleEdgePlane);
        
        // Clamp to the bounds of the line segment
        intersectT = min(1.0f, intersectT);
        intersectT = max(0.0f, intersectT);

        float dist = (e - (edge.pointAt(intersectT))).norm();

        rmin = min(dist, rmin);
        rmax = max(dist, rmax);
    }

    Eigen::AlignedBox<float, 2> bounds(
            Eigen::Vector2f(0, 0), 
            Eigen::Vector2f(imgWidth, imgHeight));

    if (bounds.contains(e)) {
        rmin = 0;
    }
}

