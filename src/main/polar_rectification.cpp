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

    int maxRValues = radMax - radMin + 0.5f;

    rectified.resize(maxRValues, numEpipoles, 1, original.spectrum());

    reverseMap.resize(maxRValues, numEpipoles, 2);

    // It's possible that the resulting transform will result in a reflection.
    // This would be problematic for computing stereo with conventional
    // algorithms, so we must detect and correct for it.
    float rFactor = 1;
    float rStart = radMin;

    if (numEpipoles > 2) {
        // if the epipolar lines go in clockwise order, then reflect
        const auto& eLine0 = epipoleLines[0][imgId];
        const auto& eLine1 = epipoleLines[1][imgId];

        float cross =
            eLine0.x() * eLine1.y() -
            eLine0.y() * eLine1.x();

        if (cross < 0) {
            rFactor = -1;
            rStart = radMax;
        }
    }

    for (int eI = 0; eI < numEpipoles; eI++) {
        const array<Eigen::Vector2f, 2>& endpoint = epipoleLines[eI];

        Eigen::Vector2f eLineDir =  endpoint[imgId];

        Eigen::ParametrizedLine<float, 2> eLine(e, eLineDir);

        cimg_forC(original, c) {
            for (int r = 0; r < maxRValues; r++) {
                Eigen::Vector2f pt = eLine.pointAt(r * rFactor + rStart);

                float out;

                rectified(r, eI, 0, c) =
                    original.linear_atXY(pt.x(), pt.y(), 0, c, out);

                reverseMap(r, eI, 0) = pt.x();

                reverseMap(r, eI, 1) = pt.y();
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
        if (rel[(startIndex + i) % 4]) {
            edges.push_back(allEdges[(startIndex + i) % 4]);
        }
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

    Eigen::Vector2f l0;

    // Procede to compute lines as if we are given a point in image 0.

    l0 = (originalPt - e0).normalized();

    Eigen::Vector3f line1DirH = fundMat * originalPt.homogeneous();
    line = line1DirH.hnormalized().unitOrthogonal();

    // Project the known match onto each line
    float line0MatchProj = (match0 - e0).normalized().dot(l0);

    float line1MatchProj = (match1 - e1).normalized().dot(line);

    // If the match point's epipolar line is nearly orthogonal to this
    // epipolar line, then we can't reliably determine which side is which.
    // So, we can rotate the matched point 90 degrees.  Note that this
    // assumes that the input images to rectify are facing the same direction.
    if (fabs(line0MatchProj) < 0.25f) {
        line0MatchProj = (match0 - e0).unitOrthogonal().dot(l0);
        line1MatchProj = (match1 - e1).unitOrthogonal().dot(line);
    }

    // Flip the direction of line 1 if necessary to select the correct
    // half-epipolar line.
    if (line0MatchProj > 0 != line1MatchProj > 0) {
        line *= -1;
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
            bool nonNeg = eLines[j].dot(candidatePlane) >= 0;

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

    assert(curCandidate == 2);

    return true;
}

void PolarRectification::createRectificationMap() {
    epipoleLines.clear();

    vector<Eigen::ParametrizedLine<float, 2>> edges0;
    vector<Eigen::ParametrizedLine<float, 2>> edges1;

    getRelevantEdges(0, edges0);
    getRelevantEdges(1, edges1);

    Eigen::AlignedBox<float, 2> imgBounds(
            Eigen::Vector2f(0, 0), 
            Eigen::Vector2f(imgWidth, imgHeight));

    // Determine the start and end vectors...

    // Use image 0 to determine the start and end directions
    Eigen::Vector2f start0Dir;
    Eigen::Vector2f end0Dir;

    {
        const auto startPoint = edges0[0].pointAt(0.0f);
        const auto endPoint = edges0[edges0.size() - 1].pointAt(1.0f);

        start0Dir = (startPoint - epipoles[0]).normalized();
        end0Dir = (endPoint - epipoles[0]).normalized();

        // Force counter-clockwise ordering
        float cross =
            start0Dir.x() * end0Dir.y() -
            start0Dir.y() * end0Dir.x();

        if (cross < 0) {
            swap(start0Dir, end0Dir);
        }
    }

    // Use image 1 to determine the start and end directions
    Eigen::Vector2f start1Dir;
    Eigen::Vector2f end1Dir;

    {
        const auto startPoint = edges1[0].pointAt(0.0f);
        const auto endPoint = edges1[edges1.size() - 1].pointAt(1.0f);

        getEpipolarLine(1, startPoint, start1Dir);
        getEpipolarLine(1, endPoint, end1Dir);
        
        // Force counter-clockwise ordering
        float cross =
            start1Dir.x() * end1Dir.y() -
            start1Dir.y() * end1Dir.x();

        if (cross < 0) {
            swap(start1Dir, end1Dir);
        }
    }

    // Find the intersection of the arcs defined by the above direction vector
    // pairs
    Eigen::Vector2f startDir;
    Eigen::Vector2f endDir;

    if (imgBounds.contains(epipoles[0]) &&
            imgBounds.contains(epipoles[1])) {
        printf("Epipoles inside image 0 & image 1\n");
        // If both the epipole passes through both images, the start and end
        // direction should be the same.  We can arbitrarily choose one.
        startDir = Eigen::Vector2f(1, 0);
        endDir = Eigen::Vector2f(1, 0);
    } else if (imgBounds.contains(epipoles[1])) {
        printf("Epipoles inside image 1\n");
        startDir = start0Dir;
        endDir = end0Dir;
    } else if (imgBounds.contains(epipoles[0])) {
        printf("Epipoles inside image 0\n");
        startDir = start1Dir;
        endDir = end1Dir;
    } else {
        // If neither epipole is within the image, we must compute intersection.
        // Note that this not handle the case where the intersection is empty
        // since for the fundamental matrix to have been computed, we must
        // have some non-empty common region between the rectified images.
        float startCross =
            start0Dir.x() * start1Dir.y() -
            start0Dir.y() * start1Dir.x();

        startDir = (startCross < 0) ? start0Dir : start1Dir;

        float endCross =
            end0Dir.x() * end1Dir.y() -
            end0Dir.y() * end1Dir.x();

        endDir = (endCross > 0) ? end0Dir : end1Dir;
    }

    // The direction of the current epipolar line
    // To begin, this should be along the direction of the first clipping-plane
    Eigen::Vector2f curLineVec = startDir;

    int maxSteps = 2 * (imgWidth * 2 + imgHeight * 2);

    for (int counter = 0; counter < maxSteps; counter++) {
        // Compute the intersection of the current epipolar line in image 0 with a
        // relevant edge by casting a ray from the epipole.
        Eigen::ParametrizedLine<float, 2> curLine0(epipoles[0], curLineVec);

        float closestIntersectionParam0 = std::numeric_limits<float>::max();

        for (const auto& edge : edges0) {
            Eigen::Hyperplane<float, 2> edgePlane(edge);
            float iParam = curLine0.intersectionParameter(edgePlane);

            if (iParam > 0) {
                closestIntersectionParam0 = min(closestIntersectionParam0,
                        iParam);
            }
        }

        Eigen::Vector2f curLineEndpointImg0 = curLine0.pointAt(closestIntersectionParam0);

        // Compute the intersection of the current epipolar line in image 1 with a
        // relevant edge by casting a ray from the epipole.
        Eigen::Vector2f curLineVec1;

        getEpipolarLine(0, curLineEndpointImg0, curLineVec1);

        Eigen::ParametrizedLine<float, 2> curLine1(epipoles[1], curLineVec1);

        float closestIntersectionParam1 = std::numeric_limits<float>::max();

        for (const auto& edge : edges1) {
            Eigen::Hyperplane<float, 2> edgePlane(edge);
            float iParam = curLine1.intersectionParameter(edgePlane);

            if (iParam > 0) {
                closestIntersectionParam1 = min(closestIntersectionParam1,
                        iParam);
            }
        }

        Eigen::Vector2f curLineEndpointImg1 = curLine1.pointAt(closestIntersectionParam1);

        // A point on next epipolar line, using image 0, in image 0
        Eigen::Vector2f nextLine0Pt0 = curLineEndpointImg0 + curLineVec.unitOrthogonal();
        Eigen::Vector2f nextLine0Dir0 = (nextLine0Pt0 - epipoles[0]).normalized();
        
        // Map nextLine0 into image 1
        Eigen::Vector2f nextLine0Dir1;
        getEpipolarLine(0, nextLine0Pt0, nextLine0Dir1);
       
        // A point on next epipolar line, using image 1, in image 1...
        
        // First, we must compute the direction to move in image 1
        Eigen::Vector2f arcDirImg1 = 
            (nextLine0Dir1 -
                (nextLine0Dir1.dot(curLineVec1) * curLineVec1)).normalized();

        Eigen::Vector2f nextLine1Pt1 = curLineEndpointImg1 + arcDirImg1;
        Eigen::Vector2f nextLine1Dir1 = (nextLine1Pt1 - epipoles[1]).normalized();
        
        // Map nextLine1 into image 0
        Eigen::Vector2f nextLine1Dir0;
        getEpipolarLine(1, nextLine1Pt1, nextLine1Dir0);

        // Use the next epipolar line which is closest
        float dist0 = (nextLine0Dir0 - curLineVec).squaredNorm();
        float dist1 = (nextLine1Dir0 - curLineVec).squaredNorm();

        Eigen::Vector2f& next = (dist0 < dist1) ? nextLine0Dir0 : nextLine1Dir0;

        // If the vector crosses over the end direction, then stop iterating.
        if (
                Eigen::Vector3f(curLineVec.x(), curLineVec.y(), 0).cross(
                    Eigen::Vector3f(endDir.x(), endDir.y(), 0)).z() > 0 &&
                Eigen::Vector3f(next.x(), next.y(), 0).cross(
                    Eigen::Vector3f(endDir.x(), endDir.y(), 0)).z() <= 0) {
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

        dist = (corner0 - e).norm();

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

