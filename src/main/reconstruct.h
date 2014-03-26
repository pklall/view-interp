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
        typedef array<double, 6> CameraParam;

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

                    p[0] += camera_translation[0];
                    p[1] += camera_translation[1];
                    p[2] += camera_translation[2];

                    T prot[3];

                    ceres::AngleAxisRotatePoint(camera_rotation, p.data(),
                            prot);

                    T predicted_x = prot[0] / prot[2];
                    T predicted_y = prot[1] / prot[2];
                    
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

        inline void addObservation(
                int cameraIndex,
                int pointIndex,
                const Eigen::Vector2d& point) {
            ceres::CostFunction* costFunction =
                new ceres::AutoDiffCostFunction<
                // 2 residuals
                // 3 parameters in block 1 (rotation)
                // 3 parameters in block 2 (translation)
                // 1 parameter in block 3 (depth)
                ReprojectionError, 2, 3, 3, 1>(
                        new ReprojectionError(
                            (double) points[pointIndex][0],
                            (double) points[pointIndex][1],
                            (double) point[0],
                            (double) point[1]));

            problem->AddResidualBlock(
                    costFunction,
                    lossFunc.get(),
                    cameras[cameraIndex].data(),
                    cameras[cameraIndex].data() + 3,
                    &(points[pointIndex][2]));
        }

        void solve();

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
