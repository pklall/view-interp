#include "polar_rectification.h"

bool PolarRectification::init(
        int _width,
        int _height,
        Eigen::Matrix3d _F,
        array<Eigen::Vector2d, 2> _match) {
    imgWidth = _width;
    imgHeight = _height;
    F = _F;
    match = _match;

    Eigen::Vector3d epipole0 = F.fullPivLu().kernel().col(0);
    Eigen::Vector3d epipole1 = F.transpose().fullPivLu().kernel().col(0);

    epipoles[0] = epipole0.hnormalized();
    epipoles[1] = epipole1.hnormalized();

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (!isfinite(epipoles[i][j])) {
                return false;
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        matchDir[i] = (_match[i] - epipoles[i]).normalized();
    }

    detectReflection();

    createRectificationMap();

    return true;
}

void PolarRectification::rectify(
        int imgId,
        const CImg<uint8_t>& original,
        CImg<uint8_t>& rectified,
        CImg<float>& reverseMap) const {
    double radMin, radMax;

    getEpipolarDistanceRanges(imgId, radMin, radMax);

    const int numEpipoles = epipoleLines.size();

    const Eigen::Vector2d& e = epipoles[imgId];

    int maxRValues = radMax - radMin + 0.5f;

    rectified.resize(maxRValues, numEpipoles, 1, original.spectrum());

    reverseMap.resize(maxRValues, numEpipoles, 2);

    // It's possible that the resulting transform will result in a reflection.
    // This would be problematic for computing stereo with conventional
    // algorithms, so we must detect and correct for it.
    double rFactor = 1;
    double rStart = radMin;

    if (epipolesReflected && imgId == 1) {
        rFactor = -1;
        rStart = radMax;
    }

    for (int eI = 0; eI < numEpipoles; eI++) {
        const Eigen::Vector2d& eLineDir = epipoleLines[eI].direction[imgId];

        Eigen::ParametrizedLine<double, 2> eLine(e, eLineDir);

        cimg_forC(original, c) {
            for (int r = 0; r < maxRValues; r++) {
                Eigen::Vector2d pt = eLine.pointAt(r * rFactor + rStart);

                float out;

                rectified(r, eI, 0, c) =
                    original.linear_atXY(pt.x(), pt.y(), 0, c, out);

                reverseMap(r, eI, 0) = pt.x();

                reverseMap(r, eI, 1) = pt.y();
            }
        }
    }
}

/*
void PolarRectification::rectifyBlock(
        int startRow,
        const CImg<uint8_t>& original0,
        const CImg<uint8_t>& original1,
        int vertSampleFactor,
        int horSampleFactor,
        int maxPixelCount,
        CImg<uint8_t>& rectified0,
        CImg<uint8_t>& rectified1,
        float& rectificationDisparity,
        int& nextRow,
        int& remainingRowCount) const {
    float radMin, radMax;

    getEpipolarDistanceRanges(imgId, radMin, radMax);

    const int numEpipoles = epipoleLines.size();

    const Eigen::Vector2d& e = epipoles[imgId];

    int maxRValues = radMax - radMin + 0.5f;

    rectified.resize(maxRValues, numEpipoles, 1, original.spectrum());

    reverseMap.resize(maxRValues, numEpipoles, 2);

    // It's possible that the resulting transform will result in a reflection.
    // This would be problematic for computing stereo with conventional
    // algorithms, so we must detect and correct for it.
    float rFactor = 1;
    float rStart = radMin;

    if (epipolesReflected) {
        // if the epipolar lines go in clockwise order, then reflect
        const auto& eLine0 = epipoleLines[0].direction[imgId];
        const auto& eLine1 = epipoleLines[1].direction[imgId];

        float cross =
            eLine0.x() * eLine1.y() -
            eLine0.y() * eLine1.x();

        if (cross < 0) {
            rFactor = -1;
            rStart = radMax;
        }
    }

    for (int eI = 0; eI < numEpipoles; eI++) {
        const Eigen::Vector2d& eLineDir = epipoleLines[eI].direction[imgId];

        Eigen::ParametrizedLine<float, 2> eLine(e, eLineDir);

        cimg_forC(original, c) {
            for (int r = 0; r < maxRValues; r++) {
                Eigen::Vector2d pt = eLine.pointAt(r * rFactor + rStart);

                float out;

                rectified(r, eI, 0, c) =
                    original.linear_atXY(pt.x(), pt.y(), 0, c, out);

                reverseMap(r, eI, 0) = pt.x();

                reverseMap(r, eI, 1) = pt.y();
            }
        }
    }
}
*/

void PolarRectification::getRelevantEdges(
        int imgId,
        vector<Eigen::ParametrizedLine<double, 2>>& edges) const {
    const auto& epipole = epipoles[imgId];

    // All 4 lines are parameterized in counter-clockwise order: top, left,
    // right, bottom.
    array<Eigen::ParametrizedLine<double, 2>, 4> allEdges;

    // top edge facing left
    allEdges[0] = Eigen::ParametrizedLine<double, 2>(
            Eigen::Vector2d(imgWidth, 0),
            Eigen::Vector2d(-imgWidth, 0));

    // left edge facing down
    allEdges[1] = Eigen::ParametrizedLine<double, 2>(
            Eigen::Vector2d(0, 0),
            Eigen::Vector2d(0, imgHeight));

    // bottom edge facing right
    allEdges[2] = Eigen::ParametrizedLine<double, 2>(
            Eigen::Vector2d(0, imgHeight),
            Eigen::Vector2d(imgWidth, 0));
    
    // right edge facing up
    allEdges[3]  = Eigen::ParametrizedLine<double, 2>(
            Eigen::Vector2d(imgWidth, imgHeight),
            Eigen::Vector2d(0, -imgHeight));

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
        const Eigen::Vector2d& originalPt,
        Eigen::Vector2d& line) const {
    // Flip the problem when necessary.
    const auto& e0 = (imgId == 0) ? epipoles[0] : epipoles[1];

    Eigen::Matrix3d fundMat;

    if (imgId == 0) {
        fundMat = F;
    } else {
        fundMat = F.transpose();
    }

    const auto& match0 = (imgId == 0) ? matchDir[0] : matchDir[1];
    const auto& match1 = (imgId == 0) ? matchDir[1] : matchDir[0];

    Eigen::Vector2d l0;

    // Procede to compute lines as if we are given a point in image 0.

    l0 = (originalPt - e0).normalized();

    Eigen::Vector3d line1DirH = fundMat * originalPt.homogeneous();
    line = line1DirH.hnormalized().unitOrthogonal();

    // Project the known match onto each line
    double line0MatchProj = match0.dot(l0);

    double line1MatchProj = match1.dot(line);

    // If the match point's epipolar line is nearly orthogonal to this
    // epipolar line, then we can't reliably determine which side is which.
    // So, we can rotate the matched point 90 degrees.
    if (fabs(line0MatchProj) < 0.25f) {
        Eigen::Vector2d match0R(-match0.y(), match0.x());
        Eigen::Vector2d match1R(-match1.y(), match1.x());

        if (epipolesReflected) {
            match1R *= -1;
        }

        line0MatchProj = match0R.dot(l0);

        line1MatchProj = match1R.dot(line);
    }

    // Flip the direction of line 1 if necessary to select the correct
    // half-epipolar line.
    if (line0MatchProj > 0 != line1MatchProj > 0) {
        line *= -1;
    }
}

void PolarRectification::getEpipolarLine(
        int imgId,
        const Eigen::Vector2d& originalPt,
        Eigen::Hyperplane<double, 2>& plane) const {
    Eigen::Vector2d lineDir;

    getEpipolarLine(imgId, originalPt, lineDir);

    plane = Eigen::Hyperplane<double, 2>(
            lineDir.unitOrthogonal(),
            epipoles[(imgId + 1) % 2]);
}

bool PolarRectification::getImg0ClippingPlanes(
        array<Eigen::Vector2d, 2>& planes) const {
    Eigen::AlignedBox<double, 2> img1(
            Eigen::Vector2d(0, 0), 
            Eigen::Vector2d(imgWidth, imgHeight));

    if (img1.contains(epipoles[1])) {
        return false;
    }

    array<Eigen::Vector2d, 4> eLines;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2d corner;

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
        Eigen::Vector2d candidatePlane = eLines[i].unitOrthogonal();

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

void PolarRectification::getEpipoleDistanceRange(
        int imgId,
        const Eigen::Vector2d& direction,
        double& rmin,
        double& rmax) const {
    // Corners in counter-clockwise order
    Eigen::Matrix<double, 2, 4> corners;
    corners <<
        0, imgWidth, imgWidth,  0,
        0, 0,        imgHeight, imgHeight;

    Eigen::ParametrizedLine<double, 2> eLine(epipoles[imgId], direction);

    rmin = std::numeric_limits<double>::min();
    rmax = std::numeric_limits<double>::max();

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2d corner0 = corners.col(i);
        Eigen::Vector2d corner1 = corners.col((i + 1) % 4);

        // Note that, since corners are enumerated in ccw order, the normal
        // faces *inside* the image.
        Eigen::Hyperplane<double, 2> edgePlane(
                (corner1 - corner0).unitOrthogonal(), corner1);

        double intersectT = eLine.intersectionParameter(edgePlane);

        // Ignore backwards intersections
        if (intersectT > 0) {
            if (direction.dot(edgePlane.normal()) < 0) {
                // The maximum radius should be the minimum of distances along the
                // line to planes facing the epipole
                rmax = min(rmax, intersectT);
            } else {
                rmin = max(rmin, intersectT);
            }
        }
    }

    // Minimum radius cannot be negative
    rmin = max(0.0, rmin);
}

void PolarRectification::detectReflection() {
    // If both epipoles are within the images, there cannot be any reflection.
    Eigen::AlignedBox<double, 2> imgBounds(
            Eigen::Vector2d(0, 0), 
            Eigen::Vector2d(imgWidth, imgHeight));

    if (imgBounds.contains(epipoles[0]) &&
            imgBounds.contains(epipoles[1])) {
        epipolesReflected = false;

        return;
    }

    /**
     * The magic below is based on the following:
     *  - Let (x, y) be a point to the right of the epipole.
     *  - A = [x, y, 1]'
     *  - B = [x, y + eps, 1]'
     *  - So, vector AB is oriented counter-clockwise relative to the epipole.
     *  - A_tH = F * A
     *  - B_tH = F * B
     *  - A_t = [A[0], A[1]]'
     *  - B_t = [B[0], B[1]]'
     *  - Now, A_t and B_t are scaled normals to the epipolar lines in image 1
     *    which correspond to ponits A and B in image 0.
     *    We should expect A_t x B_t to be positive iff these normals are rotating
     *    counter-clockwise for small eps.
     *  - Using maxima to symbolically differentiate W.R.T eps and evaluate
     *    at 0, we get the result below, which has been split into positive
     *    and negative terms to increase numerical stability.
     */
    double x = imgWidth + epipoles[0].x();

    double lhs = F(0, 1) * (F(1, 0) * x + F(1, 2));
    double rhs = F(1, 1) * (F(0, 0) * x + F(0, 2));

    epipolesReflected = lhs > rhs;
} 

void PolarRectification::createRectificationMap() {
    epipoleLines.clear();

    vector<Eigen::ParametrizedLine<double, 2>> edges0;
    vector<Eigen::ParametrizedLine<double, 2>> edges1;

    getRelevantEdges(0, edges0);
    getRelevantEdges(1, edges1);

    Eigen::AlignedBox<double, 2> imgBounds(
            Eigen::Vector2d(0, 0), 
            Eigen::Vector2d(imgWidth, imgHeight));

    // Determine the start and end vectors...

    // Use image 0 to determine the start and end directions
    Eigen::Vector2d start0Dir;
    Eigen::Vector2d end0Dir;

    {
        const auto startPoint = edges0[0].pointAt(0.0);
        const auto endPoint = edges0[edges0.size() - 1].pointAt(1.0);

        start0Dir = (startPoint - epipoles[0]).normalized();
        end0Dir = (endPoint - epipoles[0]).normalized();

        // Force counter-clockwise ordering
        double cross =
            start0Dir.x() * end0Dir.y() -
            start0Dir.y() * end0Dir.x();

        if (cross < 0) {
            swap(start0Dir, end0Dir);
        }
    }

    // Use image 1 to determine the start and end directions
    Eigen::Vector2d start1Dir;
    Eigen::Vector2d end1Dir;

    {
        const auto startPoint = edges1[0].pointAt(0.0);
        const auto endPoint = edges1[edges1.size() - 1].pointAt(1.0);

        getEpipolarLine(1, startPoint, start1Dir);
        getEpipolarLine(1, endPoint, end1Dir);
        
        // Force counter-clockwise ordering
        double cross =
            start1Dir.x() * end1Dir.y() -
            start1Dir.y() * end1Dir.x();

        if (cross < 0) {
            swap(start1Dir, end1Dir);
        }
    }

    // Find the intersection of the arcs defined by the above direction vector
    // pairs.  This will define the feasible region of the rectified images.
    Eigen::Vector2d startDir;
    Eigen::Vector2d endDir;

    if (imgBounds.contains(epipoles[0]) &&
            imgBounds.contains(epipoles[1])) {
        printf("Epipoles inside image 0 & image 1\n");
        // If both the epipole passes through both images, the start and end
        // direction should be the same.  We can arbitrarily choose one.
        startDir = Eigen::Vector2d(1, 0);
        endDir = Eigen::Vector2d(1, 0);
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
        double startCross =
            start0Dir.x() * start1Dir.y() -
            start0Dir.y() * start1Dir.x();

        startDir = (startCross < 0) ? start0Dir : start1Dir;

        double endCross =
            end0Dir.x() * end1Dir.y() -
            end0Dir.y() * end1Dir.x();

        endDir = (endCross > 0) ? end0Dir : end1Dir;
    }

    // The direction of the current epipolar line
    // To begin, this should be along the direction of the first clipping-plane
    Eigen::Vector2d curLineVec = startDir;

    int maxSteps = 2 * (imgWidth * 2 + imgHeight * 2);

    for (int counter = 0; counter < maxSteps; counter++) {
        EpipolarLineSample curSample;

        curSample.direction[0] = curLineVec;

        // Compute the intersection of the current epipolar line in image 0 with a
        // relevant edge by casting a ray from the epipole.
        Eigen::ParametrizedLine<double, 2> curLine0(epipoles[0], curLineVec);

        getEpipoleDistanceRange(0, curLineVec, curSample.minRadius[0],
                curSample.maxRadius[0]);

        Eigen::Vector2d curLineEndpointImg0 = curLine0.pointAt(curSample.maxRadius[0]);

        // Compute the intersection of the current epipolar line in image 1 with a
        // relevant edge by casting a ray from the epipole.
        Eigen::Vector2d curLineVec1;

        getEpipolarLine(0, curLineEndpointImg0, curLineVec1);

        curSample.direction[1] = curLineVec1;

        Eigen::ParametrizedLine<double, 2> curLine1(epipoles[1], curLineVec1);

        getEpipoleDistanceRange(1, curLineVec1, curSample.minRadius[1],
                curSample.maxRadius[1]);

        Eigen::Vector2d curLineEndpointImg1 = curLine1.pointAt(curSample.maxRadius[1]);

        // Push the current epipolar line
        epipoleLines.push_back(curSample);

        // Compute the next epipolar line by considering those resulting from moving
        // one unit along the edge of image 0 and image 1 from the current epipolar
        // line.

        // A point on next epipolar line, using image 0, in image 0
        Eigen::Vector2d nextLine0Pt0 = curLineEndpointImg0 + curLineVec.unitOrthogonal();
        Eigen::Vector2d nextLine0Dir0 = (nextLine0Pt0 - epipoles[0]).normalized();
        
        // Map nextLine0 into image 1
        Eigen::Vector2d nextLine0Dir1;
        getEpipolarLine(0, nextLine0Pt0, nextLine0Dir1);
       
        // A point on next epipolar line, using image 1, in image 1...
        
        // First, we must compute the direction to move in image 1
        Eigen::Vector2d arcDirImg1 = 
            (nextLine0Dir1 -
                (nextLine0Dir1.dot(curLineVec1) * curLineVec1)).normalized();

        Eigen::Vector2d nextLine1Pt1 = curLineEndpointImg1 + arcDirImg1;
        
        // Map nextLine1 into image 0
        Eigen::Vector2d nextLine1Dir0;
        getEpipolarLine(1, nextLine1Pt1, nextLine1Dir0);

        // Use the next epipolar line which is closest
        double dist0 = (nextLine0Dir0 - curLineVec).squaredNorm();
        double dist1 = (nextLine1Dir0 - curLineVec).squaredNorm();

        Eigen::Vector2d& next = (dist0 <= dist1) ? nextLine0Dir0 : nextLine1Dir0;

        // If the vector crosses over the end direction, then stop iterating.
        if (Eigen::Vector3d(curLineVec.x(), curLineVec.y(), 0).cross(
                    Eigen::Vector3d(endDir.x(), endDir.y(), 0)).z() > 0 &&
                Eigen::Vector3d(next.x(), next.y(), 0).cross(
                    Eigen::Vector3d(endDir.x(), endDir.y(), 0)).z() <= 0) {
            break;
        }

        curLineVec = next;
    }
}

void PolarRectification::getEpipolarDistanceRanges(
        int imgId,
        double& rmin,
        double& rmax) const {
    const Eigen::Vector2d& e = epipoles[imgId];

    rmin = std::numeric_limits<double>::max();
    rmax = std::numeric_limits<double>::min();

    // Corners in counter-clockwise order
    Eigen::Matrix<double, 2, 4> corners;
    corners <<
        0, imgWidth, imgWidth,  0,
        0, 0,        imgHeight, imgHeight;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2d corner0 = corners.col(i);
        Eigen::Vector2d corner1 = corners.col((i + 1) % 4);

        Eigen::ParametrizedLine<double, 2> edge(corner0, corner1 - corner0);

        Eigen::Hyperplane<double, 2> epipoleEdgePlane(
                (corner1 - corner0).normalized(), e);

        // This is the parameter associated with the closest point on the
        // edge to the epipole.
        double intersectT = edge.intersectionParameter(epipoleEdgePlane);
        
        // Clamp to the bounds of the line segment
        intersectT = min(1.0, intersectT);
        intersectT = max(0.0, intersectT);

        double dist = (e - (edge.pointAt(intersectT))).norm();

        rmin = min(dist, rmin);

        dist = (corner0 - e).norm();

        rmax = max(dist, rmax);
    }

    Eigen::AlignedBox<double, 2> bounds(
            Eigen::Vector2d(0, 0), 
            Eigen::Vector2d(imgWidth, imgHeight));

    if (bounds.contains(e)) {
        rmin = 0;
    }
}

