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

void PolarRectification::getEpipolarLine(
        int imgId,
        const Eigen::Vector2f& originalPt,
        Eigen::Hyperplane<float, 2>& plane) const {
    Eigen::Vector2f lineDir;

    getEpipolarLine(imgId, originalPt, lineDir);

    plane = Eigen::Hyperplane<float, 2>(
            lineDir.unitOrthogonal(),
            epipoles[(imgId + 1) % 2]);
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
        vector<array<Eigen::Vector2f, 2>>& endpoints) const {
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
            // Each step should move a maximum of one pixel along the edge of
            // each image.
            //
            // Thus, we must consider 3 epipolar lines and their intersection
            // with the boundaries of image 0 and image 1:
            //  - The current epipolar line (cur)
            //  - The epipolar line formed by rotating the current line as little
            //    as possible without shrinking pixels in image 0. (maxStep0)
            //  - The epipolar line formed by rotating the current line as little
            //    as possible without shrinking pixels in image 1. (maxStep1)
            
            // The current epipolar line intersection in each image
            Eigen::Vector2f curImg0Pt;
            Eigen::Vector2f curImg1Pt;

            // The points of intersection of the epipolar lines associated with
            // the maximum step with image boundaries.
            Eigen::Vector2f maxStep0Img0Pt;
            Eigen::Vector2f maxStep0Img1Pt;
            Eigen::Vector2f maxStep1Img0Pt;
            Eigen::Vector2f maxStep1Img1Pt;
            
            // The line parameters associated with the above points
            float curImg0T;
            float curImg1T;
            float maxStep0Img0T;
            float maxStep1Img0T;
            float maxStep0Img1T;
            float maxStep1Img1T;

            Eigen::Hyperplane<float, 2> curImg1Plane;
            Eigen::Hyperplane<float, 2> maxStep0Img1Plane;
            Eigen::Hyperplane<float, 2> maxStep1Img0Plane;

            curImg0T = curT;
            curImg0Pt = edge.pointAt(curImg0T);
            getEpipolarLine(0, curImg0Pt, curImg1Plane);

            maxStep0Img0T = min(curT + maxStep, 1.0f);
            maxStep0Img0Pt = edge.pointAt(maxStep0Img0T);
            getEpipolarLine(0, maxStep0Img0Pt, maxStep0Img1Plane);

            // Find the relevant edge in image 1 which intersects curPtL1
            Eigen::ParametrizedLine<float, 2>* img1Edge = &(edges1[0]);

            for (int i = 0; i < edges1.size(); i++) {
                float intersectionT =
                    edges1[i].intersectionParameter(curImg1Plane);

                if (intersectionT >= 0 && intersectionT <= 1) {
                    img1Edge = &(edges1[i]);
                }
            }

            // Transfer cur to image 1
            curImg1T = img1Edge->intersectionParameter(curImg1Plane);
            curImg1Pt = img1Edge->pointAt(curImg1T);

            // Transfer maxStep0 to image 1
            maxStep0Img1T = img1Edge->intersectionParameter(maxStep0Img1Plane);
            maxStep0Img1Pt = img1Edge->pointAt(maxStep0Img1T);

            // The point along the image 1 edge corresponding to the maximum
            // allowed step.
            float maxStep1 = 1.0f /
                (img1Edge->pointAt(1) - img1Edge->pointAt(0)).norm();

            // Find maxStep1 in image 1
            maxStep1Img1T = curImg1T + copysign(maxStep1, maxStep0Img1T - curImg1T);
            maxStep1Img1T = max(min(maxStep1Img1T, 1.0f), 0.0f);
            maxStep1Img1Pt = img1Edge->pointAt(maxStep1Img1T);
            getEpipolarLine(1, maxStep1Img1Pt, maxStep1Img0Plane);

            // Transfer maxStep1 to image 0
            maxStep1Img0T = edge.intersectionParameter(maxStep1Img0Plane);
            maxStep1Img0Pt = edge.pointAt(maxStep1Img0T);

            // Choose whichever maxStep is closer (since the edge is
            // parametrized such that we step with positive T, the smaller
            // step is closer to the current epipolar line)
            if (maxStep0Img0T < maxStep0Img0T) {
                endpoints.push_back({maxStep0Img0Pt, maxStep0Img1Pt});

                curT = maxStep0Img0T;
            } else {
                endpoints.push_back({maxStep1Img0Pt, maxStep1Img1Pt});

                curT = maxStep1Img0T;
            }
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

