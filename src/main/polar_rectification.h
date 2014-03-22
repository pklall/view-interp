#pragma once

#include <Eigen/Dense>

#include "common.h"

/**
 * Wraps a fundamental matrix with functionality relevant to polar
 * rectification.
 */
class PolarFundamentalMatrix {
public:
    /**
     * Attempts to initialize with the given fundamental matrix
     * and matched point correspondence.
     *
     * If the epipolar line is parallel to either image plane, then a polar
     * representation is not possible, and the method returns false.
     */
    bool init(
            const Eigen::Matrix3d& _F,
            const  array<Eigen::Vector2d, 2>& _match);

    inline double getEpipolarDistance(
            int imgId,
            const Eigen::Vector2d& pt) {
        return (pt - getEpipole(imgId)).norm();
    }

    inline double getRadialDisparity(
            const Eigen::Vector2d& pt0,
            const Eigen::Vector2d& pt1) {
        return getEpipolarDistance(1, pt1) - getEpipolarDistance(0, pt0);
    }

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

    inline const Eigen::Vector2d& getEpipole(
            int imgId) const {
        return epipoles[imgId];
    }

    inline const array<Eigen::Vector2d, 2>& getEpipoles() const {
        return epipoles;
    }

    inline bool areEpipolesReflected() const {
        return epipolesReflected;
    }

private:

    /**
     * Initializes `epipolesReflected`.
     */
    void detectReflection();

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
     * If true, counter-clockwise rotation relative to the epipole in
     * image 0 corresponds to a clockwise rotation relative to the
     * epipole in image 1.  Thus, the resulting rectification transform
     * must perform a reflection in the radius-domain to allow proper
     * stereo estimation via conventional algorithms which do not consider
     * reflection.
     */
    bool epipolesReflected;
};

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
                const PolarFundamentalMatrix& _F);

        /**
         * Rectifies an entire image.  Note that the result may be very large
         * and include arbitrary radial offsets.  Therefore, other methods
         * below should be used instead if stereo matching is to be performed.
         */
        void rectify(
                int imgId,
                const CImg<uint8_t>& original,
                CImg<uint8_t>& rectified,
                CImg<float>& reverseMap) const;

        inline int getRectifiedSpanCount() {
            return epipoleLines.size();
        }

        /**
         * Computes the largest vertical region with the given starting
         * span index which will fit in an image buffers with the given
         * number of maximum pixels.
         *
         * Note that maximumPixelCount represents the maximum number
         * of pixels in both the left and the right images, so double
         * the memory will be required to store both rectified images.
         *
         * `endSpan` is exclusive.
         *
         * disparity[Factor|Offset] defines the linear function of disparity
         * values from horizontal offsets in the rectified images to
         * differences in radial distance from epipoles as:
         *   epipolar_distance = disparityFactor * horizontal_disparity +
         *                       disparityOffset
         */
        void maximalRectificationRange(
                int maximumPixelCount,
                int startRow,
                int& numRows,
                int& maximumWidth,
                double& disparityFactor,
                double& disparityOffset) const;

        /**
         * Evaluates the rectification transform at each sample in the 
         * relevant rectangular region of the co-domain specified by
         * startRow and numRows.
         */
        template<class Callback>
        inline void evaluateRectificationTransform(
                int imgId,
                int startRow,
                int numRows,
                Callback callback) const {
            double minR = std::numeric_limits<double>::max();
            double maxR = std::numeric_limits<double>::min();

            for (int eI = 0; eI < numRows; eI++) {
                const auto& sample = epipoleLines[startRow + eI];

                minR = min(minR, sample.minRadius[imgId]);
                maxR = max(maxR, sample.maxRadius[imgId]);
            }

            int rFactor = 1;
            double rStart = minR;

            if (F.areEpipolesReflected() && imgId == 1) {
                rFactor = -1;
                rStart = maxR;
            }

            const Eigen::Vector2d& e = F.getEpipole(imgId);

            // TODO These loops could be blocked for greater performance
            for (int eI = 0; eI < numRows; eI++) {
                const Eigen::Vector2d& eLineDir = epipoleLines[eI].direction[imgId];

                Eigen::ParametrizedLine<double, 2> eLine(e, eLineDir);

                double angle = atan2(eLineDir.y(), eLineDir.x());

                for (int r = 0; r < (maxR - minR) + 0.5; r++) {
                    double rValue = r * rFactor + rStart;

                    Eigen::Vector2i sampleDomain(r, eI);
                    Eigen::Vector2d polarDomain(angle, rValue);
                    Eigen::Vector2d range = eLine.pointAt(rValue);
                    
                    callback(sampleDomain, polarDomain, range);
                }
            }
        }

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
        PolarFundamentalMatrix F;

        /**
         * Specifies the set of epipolar lines to use for rectification.
         *
         * Each epipolar line is precisely specified by its direction vector
         * in image 0 and image 1 because they must each pass through the
         * respective epipole.
         */
        vector<EpipolarLineSample> epipoleLines;
};

class PolarStereo {
    public:
        PolarStereo(
                int maxRectificationPixels);

        void computeStereo(
                int numScales,
                float scaleStep,
                const PolarFundamentalMatrix& F,
                const CImg<uint8_t>& leftLab,
                const CImg<uint8_t>& rightLab);

        inline float disparityAt(
                int scaleIndex,
                int x,
                int y) {
            float scale = pow(scaleStep, scaleIndex);
            float xs = x * scale;
            float ys = y * scale;

            return disparityPyramid[scaleIndex].linear_atXY(xs, ys);
        }
        
    private:
        const int rectificationBufferSize;
        
        // Buffers for storing rectified segments
        array<unique_ptr<uint8_t[]>, 2> rectificationBuffers;

        float scaleStep;

        vector<CImg<float>> disparityPyramid;

        PolarRectification rectifier;
};

