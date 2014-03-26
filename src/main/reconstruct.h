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

class Reconstruction {
    private:
        typedef array<double, 10> CameraParam;

    public:
        void init(
                int numCameras,
                int numPoints,
                double robustHuberCoeff);

        inline void addObservation(
                int cameraIndex,
                int pointIndex,
                const Eigen::Vector2f point) {
            ceres::CostFunction* costFunction =
                new ceres::AutoDiffCostFunction<
                ceres::examples::SnavelyReprojectionErrorWithQuaternions, 2, 4, 6, 3>(
                        new ceres::examples::SnavelyReprojectionErrorWithQuaternions(
                            (double) point[0],
                            (double) point[1]));

            problem->AddResidualBlock(
                    costFunction,
                    lossFunc.get(),
                    cameras[cameraIndex].data(),
                    cameras[cameraIndex].data() + 4,
                    points[pointIndex].data());
        }

        void solve();

        inline const vector<Eigen::Vector3d>& getPoints() {
            return points;
        }

        inline Eigen::Vector3d getCameraSpacePoint(
                int cameraId,
                int pointId) {
            Eigen::Vector3d p;

            const double& focal = cameras[cameraId][7];
            const double& l1 = cameras[cameraId][8];
            const double& l2 = cameras[cameraId][9];

            ceres::QuaternionRotatePoint(
                    &(cameras[cameraId][0]), points[pointId].data(), p.data());

            p[0] += cameras[cameraId][4];
            p[1] += cameras[cameraId][5];
            p[2] += cameras[cameraId][6];

            double xp = - p[0] / p[2];
            double yp = - p[1] / p[2];

            // Apply second and fourth order radial distortion.
            double r2 = xp*xp + yp*yp;
            double distortion = 1.0 + r2  * (l1 + l2  * r2);

            // Compute final projected point position.
            p[0] = focal * distortion * xp;
            p[1] = focal * distortion * yp;

            return p;
        }

    private:
        vector<CameraParam> cameras;
        vector<Eigen::Vector3d> points;

        unique_ptr<ceres::LossFunction> lossFunc;
        unique_ptr<ceres::Problem> problem;
};
