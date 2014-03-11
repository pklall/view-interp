#include "polar_rectification.h"

void PolarRectification::init(
        int _width,
        int _height,
        Eigen::Matrix3f _F,
        array<Eigen::Vector2f, 2> _match) {
    imgWidth = _width;
    imgHeight = _height;
    F = _F;
    match = _match;

    epipoles[0] = F.fullPivLu().kernel().col(0).hnormalized();
    epipoles[1] = F.transpose().fullPivLu().kernel().col(0).hnormalized();

    createRectificationMap();
}

void PolarRectification::rectify(
        int imgId,
        const CImg<uint8_t>& original,
        CImg<uint8_t>& rectified,
        CImg<float>& reverseMap) const {
    float radMin, radMax;

    getEpipolarDistanceRanges(imgId, radMin, radMax);

    const int numEpipoles = epipoleLines.size();

    const Eigen::Vector2f& e = epipoles[imgId];

    rectified.resize((int) (radMax - radMin + 0.5f), numEpipoles, 1,
            original.spectrum());

    reverseMap.resize((int) (radMax - radMin + 0.5f), numEpipoles, 2);

    // Generate clipping planes
    // Note that corners are specified in counter-clockwise order
    /*
    Eigen::Matrix<float, 2, 4> corners;
    corners <<
        0, 0,         imgWidth,  imgWidth,
        0, imgHeight, imgHeight, 0;

    array<Eigen::Hyperplane<float, 2>, 4> edgePlanes;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2f corner0 = corners.col(i);
        Eigen::Vector2f corner1 = corners.col((i + 1) % 4);

        Eigen::ParametrizedLine<float, 2> edge(corner0, corner1 - corner0);

        // Note that the plane normal faces inside the image
        edgePlanes[i] = Eigen::Hyperplane<float, 2>(edge.unitOrthogonal(), corner0);
    }
    */

    for (int eI = 0; eI < epipoleLines.size(); eI++) {
        const array<Eigen::Vector2f, 2>& endpoint = epipoleLines[eI];

        Eigen::Vector2f eLineDir =  endpoint[imgId];

        Eigen::ParametrizedLine<float, 2> eLine(e, eLineDir);

        // TODO Clip the line to avoid unnecessary sampling outside the image's
        // valid region
        
        cimg_forC(original, c) {
            for (int t = radMin; t < radMax; t++) {
                Eigen::Vector2f pt = eLine.pointAt((float) t);

                rectified(t, eI, 0, c) = original.linear_atXY(pt.x(), pt.y(), 0, c);

                reverseMap(t, eI, 0) = pt.x();

                reverseMap(t, eI, 1) = pt.y();
            }
        }
    }
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

void PolarRectification::createRectificationMap() {
    epipoleLines.clear();

    vector<Eigen::ParametrizedLine<float, 2>> edges0;
    vector<Eigen::ParametrizedLine<float, 2>> edges1;

    getRelevantEdges(0, edges0);
    getRelevantEdges(1, edges1);

    array<Eigen::Vector2f, 2> img0Clip;

    bool mustClip = getImg0ClippingPlanes(img0Clip);

    if (!mustClip) {
        // Based on the termination criteria for the main loop below, this
        // should result in a single, full rotation.
        img0Clip[0] = Eigen::Vector2f(0, 1);
        img0Clip[1] = Eigen::Vector2f(0, -1);
    }

    // The direction of the current epipolar line
    // To begin, this should be along the direction of the first clipping-plane
    Eigen::Vector2f curLineVec = -1 * img0Clip[0].unitOrthogonal();

    while (true) {
        // The intersection of the current epipolar line in image 0 with a
        // relevant edge
        Eigen::Hyperplane<float, 2> curLinePlaneImg0(
                curLineVec.unitOrthogonal(), epipoles[0]);

        Eigen::Vector2f curLineEndpointImg0;
        float closestDist = std::numeric_limits<float>::max();

        for (const auto& edge : edges0) {
            Eigen::Vector2f intersect =
                edge.intersectionPoint(curLinePlaneImg0);

            float dist = (intersect - epipoles[0]).squaredNorm();

            if (dist < closestDist) {
                closestDist = dist;
                curLineEndpointImg0 = intersect;
            }
        }

        // The intersection of the current epipolar line in image 1 with a
        // relevant edge
        Eigen::Hyperplane<float, 2> curLinePlaneImg1;

        getEpipolarLine(0, curLineEndpointImg0, curLinePlaneImg1);

        Eigen::Vector2f curLineEndpointImg1;
        closestDist = std::numeric_limits<float>::max();

        for (const auto& edge : edges1) {
            Eigen::Vector2f intersect =
                edge.intersectionPoint(curLinePlaneImg1);

            float dist = (intersect - epipoles[1]).squaredNorm();

            if (dist < closestDist) {
                closestDist = dist;
                curLineEndpointImg1 = intersect;
            }
        }

        // A point on next epipolar line, using image 0, in image 0
        Eigen::Vector2f nextLine0Pt0 = curLineEndpointImg0 + curLineVec.unitOrthogonal();
        Eigen::Vector2f nextLine0Dir0 = nextLine0Pt0 - epipoles[0];
       
        // A point on next epipolar line, using image 1, in image 1
        Eigen::Vector2f nextLine1Pt1 = curLineEndpointImg1 + curLinePlaneImg1.normal();
        Eigen::Vector2f nextLine1Dir1 = nextLine1Pt1 - epipoles[1];
        
        // Map nextLine0 into image 1
        Eigen::Vector2f nextLine0Dir1;

        getEpipolarLine(0, nextLine0Pt0, nextLine0Dir1);
        
        // Map nextLine1 into image 0
        Eigen::Vector2f nextLine1Dir0;

        getEpipolarLine(0, nextLine1Pt1, nextLine1Dir0);

        // Use the next epipolar line which is closest
        float dist0 = (nextLine0Dir0 - curLineVec).squaredNorm();
        float dist1 = (nextLine1Dir0 - curLineVec).squaredNorm();

        Eigen::Vector2f& next = (dist0 < dist1) ? nextLine0Dir0 : nextLine1Dir0;

        // If the vector goes from front-to-back relative to the second clipping
        // plane, stop.
        if (curLineVec.transpose() * img0Clip[2] > 0 &&
                next.transpose() * img0Clip[2] < 0) {
            break;
        }

        curLineVec = next;

        if (dist0 < dist1) {
            epipoleLines.push_back({nextLine0Dir0, nextLine0Dir1});
        } else {
            epipoleLines.push_back({nextLine1Dir0, nextLine1Dir1});
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

