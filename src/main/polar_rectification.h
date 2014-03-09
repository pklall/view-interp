#pragma once

#include <Eigen/Dense>

#include "common.h"

class PolarRectification {
    public:
        PolarRectification(
                int _width,
                int _height,
                Eigen::Matrix3f _F,
                array<Eigen::Vector2f, 2> _match);
        

    private:
        /**
         * Relevant edges are parameterized lines, with t in [0, 1], along
         * the border of the given image such that a ray from each point
         * on the line to the epipole does not intersect any other edge.
         *
         * If the epipole is in the image, then all edges are relevant.
         * Otherwise, the edges which are furthest away are relevant.
         *
         * All edges are parameterized in counter-clockwise order, and the
         * resulting vector contains lines in head-tail order.
         */
        void getRelevantEdges(
                int imgId,
                vector<Eigen::ParametrizedLine<float, 2>>& edges) const;

        /**
         * Returns the half-epipolar lines associated with the original point
         * given relative to the image associated with imgId.
         *
         * Since each line must, by definition, intersect the epipole, only
         * the direction vector is given.
         */
        void getEpipolarLine(
                int imgId,
                const Eigen::Vector2f& originalPt,
                Eigen::Vector2f& line) const;

        /**
         * Returns clipping planes specifying the region in image 0 space
         * which maps to image 1.  Since these planes, by definition, intersect
         * epipole 0, they are specified by their normal only.
         *
         * Returns false if the epipole is inside image 1, implying that
         * the entire image 0 region is relevant.
         */
        bool getImg0ClippingPlanes(
                array<Eigen::Vector2f, 2>& planes) const;

        bool getIntersection(
                const vector<Eigen::ParametrizedLine<float, 2>>& edges,
                const Eigen::ParametrizedLine<float, 2>& line,
                int& edgeId,
                Eigen::Vector2f& intersection);

        /**
         * Generates the set of epipolar lines (in image 0) required for
         * rectification.
         * Each line is specified by epipoles[0] and an endpoint in the
         * provided array.
         *
         * \param maxPixelsPerLine The maximum number of pixels in the source
         *                         images to map to a single pixel in a 
         *                         destination image
         * \param endpoints        the output
         */
        void createRectificationMap(
                int maxPixelsPerLine,
                vector<Eigen::Vector2f>& endpoints) const;

        int imgWidth;
        int imgHeight;

        /**
         * The fundamental matrix:
         * transpose(x1) * F * x0 = 0
         */
        Eigen::Matrix3f F;

        /**
         * A single correspondence point allows orienting epipolar lines.
         */
        array<Eigen::Vector2f, 2> match;
        
        array<Eigen::Vector2f, 2> epipoles;
};

