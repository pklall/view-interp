#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "snavely_reprojection_error.h"

#include "common.h"

#include "cvutil/cvutil.h"

class ReconstructUtil {
    public:
        /**
         * Computes the camera matrix for the second camera, assuming the
         * first camera has the identity rotation and zero translation.
         *
         * Since this is ambiguous, all 4 possible candidates are returned.
         */
        static void computeCanonicalPose(
                const Eigen::Matrix3d& E,
                array<Eigen::Matrix<double, 3, 4>, 4>& candidates);

        static inline double triangulateDepth(
                const Eigen::Vector2d& pt0,
                const Eigen::Vector2d& pt1,
                const Eigen::Matrix<double, 3, 4>& P1) {
            // The following was derived by considering the correspondence
            // pt0 = (x, y) -> pt1 = (x', y')
            // and assuming the camera transformations are [I|0] and P1.
            // Consider (xi, yi, zi) = P1 * (x * depth, y * depth, depth).
            // Then, set x' = xi / zi and y' = yi / zi.
            // Solve for depth in both expansions, resulting in rational
            // expressions.  When the denominator of one of these is
            // near 0, the other should be used.

            const double& x = pt0(0);
            const double& y = pt0(1);

            double VX = P1(0, 0) * x + P1(0, 1) * y + P1(0, 2);
            double VY = P1(1, 0) * x + P1(1, 1) * y + P1(1, 2);
            double VZ = P1(2, 0) * x + P1(2, 1) * y + P1(2, 2);

            // Solution using x'
            double numX = P1(0, 3) - P1(2, 3) * pt1(0);
            double denX = VZ * pt1(0) - VX;
            // Solution using y'
            double numY = P1(0, 3) - P1(2, 3) * pt1(0);
            double denY = VZ * pt1(0) - VY;

            // Choose the better-conditioned rational expression
            if (fabs(denX) > fabs(denY)) {
                return numX / denX;
            } else {
                return numY / denY;
            }
        }
};

/**
 * Matches features in each new image to the image before it.
 * 
 * Note that feature points are transformed to image space [-1, 1] x [-1, 1].
 */
class ChainFeatureMatcher {
    private:
        float maxFeatureCount;
        float maxMatchCount;

        unique_ptr<CVFeatureMatcher> prevMatcher;
        unique_ptr<CVFeatureMatcher> curMatcher;

        // Maps points from their index in prevMatcher/curMatcher
        // to global point indices.
        unique_ptr<map<int, int>> prevPtGlobalPt;
        unique_ptr<map<int, int>> curPtGlobalPt;

        // The total number of points with at least 2 matches
        int numPoints;

        // matches[i] contains the set of matches (p, x, y) such that
        // point p appears at (x, y) in image i.
        vector<vector<tuple<int, Eigen::Vector2f>>> matches;

    public:
        inline int getNumPoints() const {
            return numPoints;
        }

        inline const vector<vector<tuple<int, Eigen::Vector2f>>>& getObservations() const {
            return matches;
        }

        ChainFeatureMatcher(
                float maxFeatureCount = 8192,
                float maxMatchCount = 4096);

        void processNext(
                const CImg<uint8_t>& gray);

        void visualizeFeatureMatches(
                function<const CImg<uint8_t>&(int)> imgLoader) const;
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

        const ChainFeatureMatcher* features;

        vector<CameraParam> cameras;

        vector<Eigen::Vector3d> points;

    public:
        ChainReconstruction(
                const ChainFeatureMatcher* features);

        void solve();

        void exportPython(
                ostream& result) const;
};

class DepthReconstruction {
    private:
        typedef tuple<Eigen::Quaterniond, Eigen::Vector3d> CameraParam;

        struct ReprojectionError {
            // (u, v): the position of the observation with respect to the image
            // center point.
            ReprojectionError(double main_x, double main_y,
                    double observed_x, double observed_y)
                : main_x(main_x), main_y(main_y),
                observed_x(observed_x), observed_y(observed_y) {}

            template <typename T>
                bool operator()(const T* const camera_rotation,
                        const T* const camera_translation,
                        const T* const depth,
                        T* residuals) const {
                    Eigen::Matrix<T, 3, 1> p;
                    p << T(main_x) * depth[0],
                      T(main_y) * depth[0],
                      depth[0];

                    p[0] -= camera_translation[0];
                    p[1] -= camera_translation[1];
                    p[2] -= camera_translation[2];

                    Eigen::Quaternion<T> rotation(
                            camera_rotation[0],
                            camera_rotation[1],
                            camera_rotation[2],
                            camera_rotation[3]);

                    p = rotation * p;

                    T predicted_x = p[0] / p[2];
                    T predicted_y = p[1] / p[2];
                    
                    // Compute final projected point position.
                    residuals[0] = predicted_x - T(observed_x);
                    residuals[1] = predicted_y - T(observed_y);

                    return true;
                }

            double main_x;
            double main_y;
            double observed_x;
            double observed_y;
        };
    public:
        void init(
                int numCameras,
                int numPoints);

        inline void setMainPoint(
                int pointIndex,
                const Eigen::Vector2d& point) {
            points[pointIndex] = Eigen::Vector3d(point[0], point[1], 1.0);
        }

        inline void setDepth(
                int pointIndex,
                double depth) {
            points[pointIndex][2] = depth;
        }

        inline void setCamera(
                int cameraIndex,
                const Eigen::Quaterniond rotation,
                const Eigen::Vector3d translation) {
            get<0>(cameras[cameraIndex]) = rotation;
            get<1>(cameras[cameraIndex]) = translation;
        }

        inline void addObservation(
                int cameraIndex,
                int pointIndex,
                const Eigen::Vector2d& point) {
            ceres::CostFunction* costFunction =
                new ceres::AutoDiffCostFunction<
                // 2 residuals
                // 4 parameters in block 1 (rotation)
                // 3 parameters in block 2 (translation)
                // 1 parameter in block 3 (depth)
                ReprojectionError, 2, 4, 3, 1>(
                        new ReprojectionError(
                            (double) points[pointIndex][0],
                            (double) points[pointIndex][1],
                            (double) point[0],
                            (double) point[1]));

            problem->AddResidualBlock(
                    costFunction,
                    lossFunc.get(),
                    get<0>(cameras[cameraIndex]).coeffs().data(),
                    get<1>(cameras[cameraIndex]).data(),
                    &(points[pointIndex][2]));
        }

        void solve(
                bool robustify);

        inline const vector<Eigen::Vector3d>& getPoints() {
            return points;
        }

    private:
        vector<CameraParam> cameras;
        /*
         * Stores the (x, y) coordates of the observation in the main image
         * and the depth associated with the point.
         *
         * Thus, the 3D point is actually (x * z, y * z, z).
         */
        vector<Eigen::Vector3d> points;

        unique_ptr<ceres::LossFunctionWrapper> lossFunc;
        unique_ptr<ceres::Problem> problem;
};
