#include "polar_rectification.h"

#include "cvutil/cvutil.h"

bool PolarFundamentalMatrix::init(
        const Eigen::Matrix3d& _F,
        const array<Eigen::Vector2d, 2>& _match) {
    F = _F;
    match = _match;

    auto Fkern = F.fullPivLu().kernel();
    auto FTkern = F.transpose().fullPivLu().kernel();

    if (Fkern.cols() < 1 ||
            FTkern.cols() < 1) {
        return false;
    }

    Eigen::Vector3d epipole0 = Fkern.col(0);
    Eigen::Vector3d epipole1 = FTkern.col(0);

    epipoles[0] = epipole0.hnormalized();
    epipoles[1] = epipole1.hnormalized();

    cout << "Epipole 0:\n" << epipoles[0] << endl;
    cout << "Epipole 1:\n" << epipoles[1] << endl;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (!isfinite(epipoles[i][j])) {
                return false;
            }
        }

        matchDir[i] = (_match[i] - epipoles[i]).normalized();
    }

    /**
     * Detect reflection...
     * Reflection implies that rotation about the epipole in one
     * image corresponds to rotation in the opposite direction
     * about the epipole in the other image.
     *
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
    double x = epipoles[0].x() + 1.0;

    double lhs = F(0, 1) * (F(1, 0) * x + F(1, 2));
    double rhs = F(1, 1) * (F(0, 0) * x + F(0, 2));

    epipolesReflected = lhs > rhs;

    return true;
} 

void PolarFundamentalMatrix::scale(
        int originalWidth,
        int originalHeight,
        int finalWidth,
        int finalHeight) {
    double scaleX = finalWidth / (double) originalWidth;
    double scaleY = finalHeight / (double) originalHeight;

    Eigen::Matrix3d invScale;
    invScale <<
        (originalWidth / (double) finalWidth), 0.0, 0.0,
        0.0, (originalHeight / (double) finalHeight), 0.0,
        0.0, 0.0, 1.0;

    Eigen::Matrix3d newF = invScale.transpose() * F * invScale;

    F = newF;

    for (int i = 0; i < 2; i++) {
        match[i] <<
            (match[i].x() * scaleX),
            (match[i].y() * scaleY);

        matchDir[i] <<
            (matchDir[i].x() * scaleX),
            (matchDir[i].y() * scaleY);

        matchDir[i].normalize();

        epipoles[i] <<
            (epipoles[i].x() * scaleX),
            (epipoles[i].y() * scaleY);
    }
}


void PolarFundamentalMatrix::getEpipolarLine(
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

void PolarFundamentalMatrix::getEpipolarLine(
        int imgId,
        const Eigen::Vector2d& originalPt,
        Eigen::Hyperplane<double, 2>& plane) const {
    Eigen::Vector2d lineDir;

    getEpipolarLine(imgId, originalPt, lineDir);

    plane = Eigen::Hyperplane<double, 2>(
            lineDir.unitOrthogonal(),
            epipoles[(imgId + 1) % 2]);
}


bool PolarRectification::init(
        int _width,
        int _height,
        const PolarFundamentalMatrix& _F) {
    imgWidth = _width;
    imgHeight = _height;
    F = _F;

    createRectificationMap();

    return true;
}

void PolarRectification::rectify(
        int imgId,
        const CImg<uint8_t>& original,
        CImg<uint8_t>& rectified,
        CImg<float>& reverseMap) const {
    int maximumPixelCount = std::numeric_limits<int>::max();
    int startRow = 0;
    int numRows;
    int maximumWidth;
    double disparityFactor, disparityOffset;

    maximalRectificationRange(maximumPixelCount, startRow, numRows,
            maximumWidth, disparityFactor, disparityOffset);

    rectified.resize(maximumWidth, numRows, 1, original.spectrum());

    reverseMap.resize(maximumWidth, numRows, 2);

    struct {
        const CImg<uint8_t>& original;
        CImg<uint8_t>& rectified;
        CImg<float>& reverseMap;
        int channel;

        void operator()(
                const Eigen::Vector2i& rCoord,
                const Eigen::Vector2d& polar,
                const Eigen::Vector2d& oCoord) {
            if (rCoord.x() >= reverseMap.width()) {
                printf("rCoord OUT OF BOUNDS = %d\n", rCoord.x());
            }
            assert(rCoord.x() < reverseMap.width());
            assert(rCoord.y() < reverseMap.height());

            reverseMap(rCoord.x(), rCoord.y(), 0) =
                oCoord.x();
            reverseMap(rCoord.x(), rCoord.y(), 1) =
                oCoord.y();

            float col = 0;

            rectified(rCoord.x(), rCoord.y(), 0, channel) =
                original.linear_atXY(oCoord.x(), oCoord.y(), 0, channel, col);
        }
    } rectCallback = {original, rectified, reverseMap, 0};

    for (int c = 0; c < original.spectrum(); c++) {
        rectCallback.channel = c;
        evaluateRectificationTransform(imgId, startRow, numRows, rectCallback);
    }
}

void PolarRectification::maximalRectificationRange(
        int maximumPixelCount,
        int startRow,
        int& numRows,
        int& maximumWidth,
        double& disparityFactor,
        double& disparityOffset) const {
    double minR[2];
    double maxR[2];

    for (int i = 0; i < 2; i++) {
        minR[i] = std::numeric_limits<double>::max();
        maxR[i] = std::numeric_limits<double>::min();
    }

    maximumWidth = 0;

    for (numRows = 0; numRows < epipoleLines.size() - startRow; numRows++) {
        for (int i = 0; i < 2; i++) {
            // Compute this before we clobber minR and maxR with values from
            // a row which may not be in the maximal span.
            if (F.areEpipolesReflected()) {
                disparityFactor = -1;
                disparityOffset = minR[0] - maxR[1];
            } else {
                disparityFactor = 1;
                disparityOffset = minR[0] - minR[1];
            }

            minR[i] = min(minR[i], epipoleLines[startRow + numRows].minRadius[i]);
            maxR[i] = max(maxR[i], epipoleLines[startRow + numRows].maxRadius[i]);

            int newMaximumWidth = (maxR[i] - minR[i]) + 2;

            if (newMaximumWidth * (numRows + 1) > maximumPixelCount) {
                numRows++;
                break;
            } else {
                maximumWidth = max(maximumWidth, newMaximumWidth);
            }
        }
    }

    if (F.areEpipolesReflected()) {
        disparityFactor = -1;
        disparityOffset = minR[0] - maxR[1];
    } else {
        disparityFactor = 1;
        disparityOffset = minR[0] - minR[1];
    }
}

void PolarRectification::getRelevantEdges(
        int imgId,
        vector<Eigen::ParametrizedLine<double, 2>>& edges) const {
    const auto& epipole = F.getEpipole(imgId);

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

bool PolarRectification::getImg0ClippingPlanes(
        array<Eigen::Vector2d, 2>& planes) const {
    Eigen::AlignedBox<double, 2> img1(
            Eigen::Vector2d(0, 0), 
            Eigen::Vector2d(imgWidth, imgHeight));

    if (img1.contains(F.getEpipole(1))) {
        return false;
    }

    array<Eigen::Vector2d, 4> eLines;

    for (int i = 0; i < 4; i++) {
        Eigen::Vector2d corner;

        // Enumerate all corners of the image in binary
        corner[0] = (i & 0x1) == 0 ? 0 : imgWidth;
        corner[1] = (i & 0x2) == 0 ? 0 : imgHeight;

        F.getEpipolarLine(1, corner, eLines[i]);
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

    Eigen::ParametrizedLine<double, 2> eLine(F.getEpipole(imgId), direction);

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

        start0Dir = (startPoint - F.getEpipole(0)).normalized();
        end0Dir = (endPoint - F.getEpipole(0)).normalized();

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

        F.getEpipolarLine(1, startPoint, start1Dir);
        F.getEpipolarLine(1, endPoint, end1Dir);
        
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

    if (imgBounds.contains(F.getEpipole(0)) &&
            imgBounds.contains(F.getEpipole(1))) {
        printf("Epipoles inside image 0 & image 1\n");
        // If both the epipole passes through both images, the start and end
        // direction should be the same.  We can arbitrarily choose one.
        startDir = Eigen::Vector2d(1, 0);
        endDir = Eigen::Vector2d(1, 0);
    } else if (imgBounds.contains(F.getEpipole(1))) {
        printf("Epipoles inside image 1\n");
        startDir = start0Dir;
        endDir = end0Dir;
    } else if (imgBounds.contains(F.getEpipole(0))) {
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
        Eigen::ParametrizedLine<double, 2> curLine0(F.getEpipole(0), curLineVec);

        getEpipoleDistanceRange(0, curLineVec, curSample.minRadius[0],
                curSample.maxRadius[0]);

        Eigen::Vector2d curLineEndpointImg0 = curLine0.pointAt(curSample.maxRadius[0]);

        // Compute the intersection of the current epipolar line in image 1 with a
        // relevant edge by casting a ray from the epipole.
        Eigen::Vector2d curLineVec1;

        F.getEpipolarLine(0, curLineEndpointImg0, curLineVec1);

        curSample.direction[1] = curLineVec1;

        Eigen::ParametrizedLine<double, 2> curLine1(F.getEpipole(1), curLineVec1);

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
        Eigen::Vector2d nextLine0Dir0 = (nextLine0Pt0 - F.getEpipole(0)).normalized();
        
        // Map nextLine0 into image 1
        Eigen::Vector2d nextLine0Dir1;
        F.getEpipolarLine(0, nextLine0Pt0, nextLine0Dir1);
       
        // A point on next epipolar line, using image 1, in image 1...
        
        // First, we must compute the direction to move in image 1
        Eigen::Vector2d arcDirImg1 = 
            (nextLine0Dir1 -
                (nextLine0Dir1.dot(curLineVec1) * curLineVec1)).normalized();

        Eigen::Vector2d nextLine1Pt1 = curLineEndpointImg1 + arcDirImg1;
        
        // Map nextLine1 into image 0
        Eigen::Vector2d nextLine1Dir0;
        F.getEpipolarLine(1, nextLine1Pt1, nextLine1Dir0);

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
    const Eigen::Vector2d& e = F.getEpipole(imgId);

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

void PolarStereo::computeStereo(
        int numScales,
        float scaleStep,
        const PolarFundamentalMatrix& F,
        const CImg<uint8_t>& leftGray,
        const CImg<uint8_t>& rightGray) {
    assert(leftGray.is_sameXYZC(rightGray));
    assert(leftGray.depth() == 1);
    assert(leftGray.spectrum() == 1);

    int imgWidth = leftGray.width();
    int imgHeight = leftGray.height();
    int imgSpectrum = leftGray.spectrum();

    // Allocate space for the resulting pyramid of disparity images
    disparityPyramid.reserve(numScales);

    int curImgWidth = imgWidth;
    int curImgHeight = imgHeight;

    PolarFundamentalMatrix curF;

    array<CImg<uint8_t>, 2> imgScaledGray;
    imgScaledGray[0] = leftGray;
    imgScaledGray[1] = rightGray;

    array<CImg<uint8_t>, 2> rectified;

    CImg<int16_t> disparity;

    disparityPyramid.resize(numScales);

    for (int i = 0; i < numScales; i++) {
        float scale = pow(scaleStep, i);
        curImgWidth = imgWidth * scale;
        curImgHeight = imgHeight * scale;

        disparityPyramid[i].resize(curImgWidth, curImgHeight, 1, 1, -1);

        for (CImg<uint8_t>& scaledImg : imgScaledGray) {
            // 3 indicates nearest-neighbor resampling
            scaledImg.resize(curImgWidth, curImgHeight, 1, imgSpectrum, 3);
        }

        curF = F;
        curF.scale(imgWidth, imgHeight, curImgWidth, curImgHeight);

        rectifier.init(curImgWidth, curImgHeight, F);

        int startRow = 0;
        int numRows;
        int maximumWidth;
        double disparityFactor, disparityOffset;

        rectifier.maximalRectificationRange(std::numeric_limits<int>::max(),
                startRow, numRows, maximumWidth, disparityFactor,
                disparityOffset);


        // Round up to the next multiple of 8 since CVStereo requires
        // a disparity range which is a multiple of 16, and the entire
        // disparity range will be [-rectifiedPadding, rectifiedPadding]
        int rectifiedPadding = ((maximumWidth + 7) / 8) * 8;

        int paddedWidth = maximumWidth + rectifiedPadding * 2;
        
        // Allocate a width 3 times as much is necessary since OpenCV's stereo
        // refuses to compute along the image border.
        for (auto& rectImg : rectified) {
            rectImg.resize(paddedWidth, numRows, 1, 1, -1);

            rectImg = (uint8_t) 0;
        }

        disparity.resize(paddedWidth, numRows, 1, 1, -1);

        disparity = -rectifiedPadding * 16;

        // Rectify...
        
        struct {
            int rectifiedPadding;
            const CImg<uint8_t>* original;
            CImg<uint8_t>* rectified;

            void operator()(
                    const Eigen::Vector2i& r,
                    const Eigen::Vector2d& p,
                    const Eigen::Vector2d& o) {
                float col = 0;

                (*rectified)(rectifiedPadding + r.x(), r.y()) =
                    original->linear_atXY(o.x(), o.y(), 0, 0, col);
            }
        } rectCallback = {rectifiedPadding,
            nullptr, nullptr};

        for (int imgId = 0; imgId < 2; imgId++) {
            rectCallback.original = &(imgScaledGray[imgId]);

            rectCallback.rectified = &(rectified[imgId]);

            rectifier.evaluateRectificationTransform(
                    imgId, startRow, numRows, rectCallback);
        }

        (rectified[0], rectified[1]).display();

        // Compute stereo
        CVStereo::stereo(
                -rectifiedPadding,
                rectifiedPadding * 2,
                paddedWidth,
                curImgHeight,
                rectified[0].data(),
                rectified[1].data(),
                disparity.data());

        // Derectify...
        struct {
            int width;
            int height;
            int rectifiedPadding;

            double disparityFactor;
            double disparityOffset;

            const CImg<int16_t>* rectified;
            CImg<float>* unrectified;

            void operator()(
                    const Eigen::Vector2i& r,
                    const Eigen::Vector2d& p,
                    const Eigen::Vector2d& o) {
                float disparity = (*rectified)(rectifiedPadding + r.x(), r.y());

                disparity /= 16.0f;

                disparity = disparity * disparityFactor + disparityOffset;

                int ox = (o.x() + 0.5);
                int oy = (o.y() + 0.5);

                if (ox < width && oy < height && ox >= 0 && oy >= 0) {
                    (*unrectified)(ox, oy) = disparity;
                }
            }
        } unrectCallback = {curImgWidth, curImgHeight, rectifiedPadding,
            disparityFactor * scale, disparityOffset,
            &disparity, &(disparityPyramid[i])};

        disparityPyramid[i] = -rectifiedPadding;

        rectifier.evaluateRectificationTransform(
                0, startRow, numRows, unrectCallback);

        disparityPyramid[i].get_equalize(256).display();
    }
}

