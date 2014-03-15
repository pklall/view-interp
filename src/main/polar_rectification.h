#pragma once

#include <Eigen/Dense>

#include "common.h"

class PolarRectification {
    public:
        /**
         * Initializes the rectification mapping for images of the given size
         * and with the given fundamental matrix and example correspondence
         * point.
         *
         * Returns true if rectification is possible (both epipoles are finite),
         * false otherwise.
         */
        bool init(
                int _width,
                int _height,
                Eigen::Matrix3d _F,
                array<Eigen::Vector2d, 2> _match);

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
                float angularSampleFactor,
                float radialSampleFactor,
                int maxPixelCount,
                CImg<uint8_t>& rectified0,
                CImg<uint8_t>& rectified1,
                float& rectificationDisparity,
                int& nextRow,
                int& remainingRowCount) const;
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
            Eigen::Vector2d direction[2];
            double minRadius[2];
            double maxRadius[2];
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
                vector<Eigen::ParametrizedLine<double , 2>>& edges) const;

        /**
         * Returns the half-epipolar lines associated with the original point
         * given relative to the image associated with imgId.
         *
         * Since each line must, by definition, intersect the epipole, only
         * the direction vector is given.
         */
        void getEpipolarLine(
                int imgId,
                const Eigen::Vector2d& originalPt,
                Eigen::Vector2d& line) const;

        void getEpipolarLine(
                int imgId,
                const Eigen::Vector2d& originalPt,
                Eigen::Hyperplane<double, 2>& plane) const;

        /**
         * Returns clipping planes specifying the region in image 0 space
         * which maps to image 1.  Since these planes, by definition, intersect
         * epipole 0, they are specified by their normal only.
         *
         * Returns false if the epipole is inside image 1, implying that
         * the entire image 0 region is relevant.
         */
        bool getImg0ClippingPlanes(
                array<Eigen::Vector2d, 2>& planes) const;

        void getEpipoleDistanceRange(
                int imgId,
                const Eigen::Vector2d& direction,
                double& rmin,
                double& rmax) const;

        /**
         * Initializes `epipolesReflected`.
         */
        void detectReflection();

        /**
         * Initializes `epipoleLines`.
         */
        void createRectificationMap();

        /**
         * Returns the minimum and maximum distance from the epipole to
         * the valid region of the specified image.
         */
        void getEpipolarDistanceRanges(
                int imgId,
                double& rmin,
                double& rmax) const;

        int imgWidth;
        int imgHeight;

        /**
         * The fundamental matrix:
         * transpose(x1) * F * x0 = 0
         */
        Eigen::Matrix3d F;

        /**
         * The location of the epipole relative to the coordinate spaces
         * of each image.
         */
        array<Eigen::Vector2d, 2> epipoles;

        /**
         * The direction, relative to each epipole, of a pair of point
         * correspondences.  This allows orienting half-epipolar lines.
         */
        array<Eigen::Vector2d, 2> match;
        array<Eigen::Vector2d, 2> matchDir;

        /**
         * Specifies the set of epipolar lines to use for rectification.
         *
         * Each epipolar line is precisely specified by its direction vector
         * in image 0 and image 1 because they must each pass through the
         * respective epipole.
         */
        vector<EpipolarLineSample> epipoleLines;

        /**
         * If true, counter-clockwise rotation relative to the epipole in
         * image 0 corresponds to a clockwise rotation relative to the
         * epipole in image 1.  Thus, the resulting rectification transform
         * must perform a reflection in the radius-domain to allow proper
         * stereo estimation via conventional algorithms which do not consider
         * reflection.
         */
        bool epipolesReflected;
};

