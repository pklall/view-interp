#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "snavely_reprojection_error.h"

#include "common.h"

#include "cvutil/cvutil.h"

/**
 * Matches features in each new image to the image before it.
 * 
 * Note that feature points are transformed to image space [-1, 1] x [-1, 1].
 */
class ChainFeatureMatcher {
    private:
        float maxFeatureCount = 4096;
        float maxMatchCount = 1024;

        unique_ptr<CVFeatureMatcher> prevMatcher;
        unique_ptr<CVFeatureMatcher> curMatcher;

        // Maps points from their index in prevMatcher/curMatcher
        // to global point indices.
        unique_ptr<map<int, int>> prevPtGlobalPt;
        unique_ptr<map<int, int>> curPtGlobalPt;

        int prevWidth;
        int prevHeight;

        // The total number of points with at least 2 matches
        int numPoints;

        // matches[i] contains the set of matches (p, x, y) such that
        // point p appears at (x, y) in image i.
        vector<vector<tuple<int, float, float>>> matches;

    public:
        inline int getNumPoints() const {
            return numPoints;
        }

        inline const vector<vector<tuple<int, float, float>>>& getObservations() const {
            return matches;
        }

        ChainFeatureMatcher();

        void processNext(
                const CImg<uint8_t>& gray);

        void visualizeFeatureMatches(
                function<const CImg<uint8_t>&(int)> imgLoader) const;
};

class ChainRectification {
    private:
        /**
         * Transforms 2D-point `original` by first performing inverse radial
         * distortion via the method of Mallon & Whelan (2004) with 6 parameters
         * from `distortion` followed by multiplication by the 3x3 matrix
         * `transform`.
         */
        template<typename T>
        static inline void transformPoint(
                const T* const distortion,
                const T* const transform,
                const T* const original,
                T* transformed) {
            typedef Eigen::Matrix<T, 3, 1> Vector3T;
            typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;

            Matrix3TConst trans(transform);

            // See Eq. 5 of Precise Radial Un-distortion of Images, Mallon &
            // Whelan, 2004.
            T rd2 = original[0] * original[0] + original[1] * original[1];
            T rd4 = rd2 * rd2;
            T rd6 = rd4 * rd2;
            T rd8 = rd6 * rd2;

            T numCoeff = 
                distortion[0] * rd2 +
                distortion[1] * rd4 +
                distortion[2] * rd6 +
                distortion[3] * rd8;

            T denom = 1 + 4 * distortion[4] * rd2 + 6 * distortion[5] * rd4;

            T pointUX = original[0] - original[0] * numCoeff / denom;
            T pointUY = original[1] - original[1] * numCoeff / denom;

            Vector3T o(pointUX, pointUY, 1);

            transformed = trans * o;
        }

        template<int NCameras>
        struct Error {
            array<Eigen::Vector2f, NCameras> points;

            template <typename T>
            bool operator()(
                    const T* const distortion,
                    const T* const transforms,
                    T* residuals) const {
                typedef Eigen::Matrix<T, 2, 1> Vector2T;

                array<Vector2T, NCameras> transformed;
                T totalY;

                for (int i = 0; i < NCameras; i++) {
                    transformPoint(
                            distortion,
                            transforms + 9 * i,
                            points[i].data(),
                            transformed[i].data());

                    totalY += transformed[i][1];
                }

                totalY /= T(NCameras);
                
                for (int i = 0; i < NCameras; i++) {
                    residuals[i] = transformed[i][1] - totalY;
                }

                return true;
            }
        };

        /**
         * Enforces unit determinant of a 3x3 matrix.
         */
        struct UnitDetPrior {
            template<typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;

                Matrix3TConst t(transform);

                residual[0] = t.determinant() - 1;
            }
        };

        struct SmallPrior {
            template<typename T>
            bool operator()(
                    const T* const param,
                    T* residual) const {
                residual[0] = param;
            }
        };

    public:
};

class ChainReconstruction {
    private:
        /**
         * A camera is parameterized by the following (in this order):
         *  - Rotation quaternion (4 parameters)
         *  - Translation (3 parameters)
         *  - Focal length (1 parameter)
         *  - Radial distortion (2 parameters)
         *
         * Note that this parameter ordering is compatible with
         * ceres-solver's SnavelyReprojectionErrorWithQuaternions.
         */
        typedef array<double, 10> CameraParam;
        typedef array<double, 3> Point3d;

        const ChainFeatureMatcher* features;

        vector<CameraParam> cameras;

        vector<Point3d> points;

    public:
        ChainReconstruction(
                const ChainFeatureMatcher* features);

        void solve();

        void exportPython(
                ostream& result) const;
};
