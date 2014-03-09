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

    // Push the relevant edges in head-to-tail order
    bool allRelevant = rel[0] && rel[1] && rel[2] && rel[3];

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

void PolarRectification::getEpipolarLines(
    int imgId,
    const Eigen::Vector2f& originalPt,
    Eigen::ParametrizedLine<float, 2>& line0,
    Eigen::ParametrizedLine<float, 2>& line1) {

    if (imgId == 0) {
        auto line0Dir = originalPt - epipoles[0];

        Eigen::Vector3f line1DirH = F * originalPt.homogeneous();
        auto line1Dir = line1DirH.hnormalized().unitOrthogonal();

        // Flip the direction of line 1 to select the correct half-epipolar line
        auto line0MatchProj =
            (match[0] - epipoles[0]).transpose() *
            line0Dir;

        auto line1MatchProj =
            (match[1] - epipoles[1]).transpose() *
            line1Dir;

        if (line0MatchProj > 0 != line1MatchProj > 0) {
            line1Dir *= -1;
        }

        line0 = Eigen::ParametrizedLine<float, 2>(epipoles[0], line0Dir);

        line1 = Eigen::ParametrizedLine<float, 2>(epipoles[1], line1Dir);
    } else {
        Eigen::Vector3f line0DirH = F.transpose() * originalPt.homogeneous();
        auto line0Dir = line0DirH.hnormalized().unitOrthogonal();

        auto line1Dir = originalPt - epipoles[1];

        // Flip the direction of line 0 to select the correct half-epipolar line
        auto line0MatchProj =
            (match[0] - epipoles[0]).transpose() *
            line0Dir;

        auto line1MatchProj =
            (match[1] - epipoles[1]).transpose() *
            line1Dir;

        if (line0MatchProj > 0 != line1MatchProj > 0) {
            line0Dir *= -1;
        }

        line0 = Eigen::ParametrizedLine<float, 2>(epipoles[0], line0Dir);

        line1 = Eigen::ParametrizedLine<float, 2>(epipoles[1], line1Dir);
    }
}

/**
 * Returns clipping planes specifying the region in image 0 space
 * which maps to image 1.
 *
 * Returns false if the epipole is inside image 1, implying that
 * the entire image 0 region is relevant.
 */
bool PolarRectification::getImg0ClippingPlanes(
        array<Eigen::Hyperplane<float, 2>, 2>& planes) {
    // FIXME
    return false;
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

    vector<Eigen::ParametrizedLine<float, 2>> edges;
}

