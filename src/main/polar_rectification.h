#pragma once

#include <Eigen/Dense>

#include "common.h"

class PolarRectification {
    public:
        void init(
                int _width,
                int _height,
                Eigen::Matrix3f _F,
                array<Eigen::Vector2f, 2> _match);

        void rectify(
                int imgId,
                const CImg<uint8_t>& original,
                CImg<uint8_t>& rectified,
                CImg<float>& reverseMap) const;

        /*
        void rectifyBlock(
                int startRow,
                const CImg<uint8_t>& original0,
                const CImg<uint8_t>& original1,
                int vertSampleFactor,
                int horSampleFactor,
                int maxPixelCount,
                CImg<uint8_t>& rectified0,
                CImg<uint8_t>& rectified1,
                float& relativeDisparity,
                int& nextRow,
                int& remainingRowCount);
        */

    private:
        /**
         * Polar rectification relies on sampling the space of all epipolar
         * half-lines through the epipole.  It is sufficient to represent
         * these by their corresponding direction vectors in each image.
         *
         * Along with each such vector, there is a relevant range of distance
         * from the epipole at which the half-line intersects each image.
         */
        struct EpipolarLineSample {
            Eigen::Vector2f direction[2];
            float minRadius[2];
            float maxRadius[2];
        };

        /**
         * Relevant edges are parameterized lines, with t in [0, 1], along
         * the border of the given image such that a ray from each point
         * on the line to the epipole does not intersect any other edge.
         *
         * This is useful because relevant edges represent one extremum of
         * image distortion during rectification.  They also provide enough
         * information to determine the relevant regions of images in which
         * stereo correspondence may actually be computed.
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

        void getEpipolarLine(
                int imgId,
                const Eigen::Vector2f& originalPt,
                Eigen::Hyperplane<float, 2>& plane) const;

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

        void getEpipoleDistanceRange(
                int imgId,
                const Eigen::Vector2f& direction,
                float& rmin,
                float& rmax) const;

        /**
         * Computes `epipoleLines`.
         */
        void createRectificationMap();

        /**
         * Returns the minimum and maximum distance from the epipole to
         * the valid region of the specified image.
         */
        void getEpipolarDistanceRanges(
                int imgId,
                float& rmin,
                float& rmax) const;

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

        /**
         * Specifies the set of epipolar lines to use for rectification.
         *
         * Each epipolar line is precisely specified by its direction vector
         * in image 0 and image 1 because they must each pass through the
         * respective epipole.
         */
        vector<EpipolarLineSample> epipoleLines;
};

