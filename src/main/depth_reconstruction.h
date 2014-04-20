#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "snavely_reprojection_error.h"

#include "common.h"

#include "cvutil/cvutil.h"

#include "polar_stereo.h"

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

        static inline void triangulate(
                const Eigen::Vector2d& pt0,
                const Eigen::Vector2d& pt1,
                const Eigen::Matrix<double, 3, 4>& P0,
                const Eigen::Matrix<double, 3, 4>& P1,
                Eigen::Vector3d& tri) {
            // See http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
            Eigen::Matrix<double, 4, 3> A;
            A <<
                (pt0.x() * P0(2, 0) - P0(0, 0)), 
                (pt0.x() * P0(2, 1) - P0(0, 1)),
                (pt0.x() * P0(2, 2) - P0(0, 2)),

                (pt0.y() * P0(2, 0) - P0(1, 0)),
                (pt0.y() * P0(2, 1) - P0(1, 1)),
                (pt0.y() * P0(2, 2) - P0(1, 2)),

                (pt1.x() * P1(2, 0) - P1(0, 0)), 
                (pt1.x() * P1(2, 1) - P1(0, 1)),
                (pt1.x() * P1(2, 2) - P1(0, 2)),

                (pt1.y() * P1(2, 0) - P1(1, 0)), 
                (pt1.y() * P1(2, 1) - P1(1, 1)),
                (pt1.y() * P1(2, 2) - P1(1, 2));

            Eigen::Vector4d B;
            B <<
                -(pt0.x() * P0(2, 3) - P0(0, 3)),
                -(pt0.y() * P0(2, 3) - P0(1, 3)),
                -(pt1.x() * P1(2, 3) - P1(0, 3)),
                -(pt1.y() * P1(2, 3) - P1(1, 3));

            Eigen::Vector3d x = A.fullPivLu().solve(B);

            tri = x;
        }

        static inline double triangulateDepth(
                const Eigen::Vector2d& pt0,
                const Eigen::Vector2d& pt1,
                const Eigen::Matrix<double, 3, 4>& P1,
                const Eigen::Matrix<double, 3, 3>& E,
                double& confidence) {
            // Depth triangulation begins by projecting pt1 onto the epipolar
            // line in image 1 associated with pt0.
            //
            // Then, we obtain an estimate of confidence based on the angle
            // between the epipolar line which pt1 was originally on and
            // the expected epipolar line on which we projected it.
            //
            // The rationale behind this estimate of confidence is that
            // large deviations far away from the epipole should be tolerated
            // since pose estimation often results in small angular differences
            // resulting in large deviation further away from the epipole.

            // pt0's epipolar line in image 1
            Eigen::Vector3d pt0E1 = E * pt0.homogeneous();

            double epipolarDist = pt1.homogeneous().dot(pt0E1);

            // The projection of pt1 onto the epipole in image 1
            Eigen::Vector2d pt1ProjPt0E1 = pt1 - epipolarDist *
                Eigen::Vector2d(pt0E1.x(), pt0E1.y());

            // pt1's epipolar line in image 0
            Eigen::Vector3d pt1E0 = E.transpose() * pt1.homogeneous();

            epipolarDist = pt0.homogeneous().dot(pt0E1);

            Eigen::Vector2d pt0ProjPt1E0 = pt0 - epipolarDist *
                Eigen::Vector2d(pt1E0.x(), pt1E0.y());

            Eigen::Vector3d pt1E1 = E * pt0ProjPt1E0.homogeneous();

            confidence = Eigen::Vector2d(pt1E1.x(), pt1E1.y()).normalized().dot(
                    Eigen::Vector2d(pt0E1.x(), pt1E1.y()).normalized());

            // FIXME
            // confidence = fabs(epipolarDist);

            // The following was derived by considering the correspondence
            // pt0 = (x, y) -> pt1 = (x', y')
            // and assuming the camera transformations are [I|0] and P1.
            // Consider (xi, yi, zi) = P1 * (x * depth, y * depth, depth).
            // Then, set x' = xi / zi and y' = yi / zi.
            // Solve for depth in both expansions, resulting in two rational
            // expressions. The average of these is then used to compute
            // depth.

            const double& x = pt0(0);
            const double& y = pt0(1);

            const double& xp = pt1ProjPt0E1(0);
            const double& yp = pt1ProjPt0E1(1);

            double VX = P1(0, 0) * x + P1(0, 1) * y + P1(0, 2);
            double VY = P1(1, 0) * x + P1(1, 1) * y + P1(1, 2);
            double VZ = P1(2, 0) * x + P1(2, 1) * y + P1(2, 2);

            // Solution using x'
            double numX = P1(0, 3) - P1(2, 3) * xp;
            double denX = VZ * xp - VX;
            // Solution using y'
            double numY = P1(1, 3) - P1(2, 3) * yp;
            double denY = VZ * yp - VY;

            // Choose the better-conditioned rational expression
            // if (fabs(denX) > fabs(denY)) {
                // return numX / denX;
            // } else {
                // return numY / denY;
            // }

            return (numX * denY + numY * denX) / (2.0 * denX * denY);
        }

        static inline int selectCandidatePose(
                const Eigen::Vector2d& pt0,
                const Eigen::Vector2d& pt1,
                const array<Eigen::Matrix<double, 3, 4>, 4>& candidates) {
            for (int i = 0; i < 4; i++) {
                Eigen::Vector3d tri;
                Eigen::Matrix<double, 3, 4> P0;
                P0 <<
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0;
                triangulate(pt0, pt1, P0, candidates[i], tri);

                if (tri.z() > 0) {
                    Eigen::Vector3d transTri = candidates[i] * tri.homogeneous();

                    if (transTri.z() > 0) {
                        return i;
                    }
                }
            }

            return -1;
        }
};

/**
 * Reconstructs a sparse depth map at keypoints matched in a predetermined
 * reference image.
 */
class DepthReconstruction {
    private:
        /**
         * Cameras are paramterized by a rotation followed by a translation.
         *
         * That is, world coordinates are transformed by the camera by
         * first rotating and *then* translating.
         */
        struct CameraParam {
            Eigen::Vector3d translation;
            Eigen::Quaterniond rotation;

            inline Eigen::Matrix<double, 3, 4> getP() {
                Eigen::Matrix3d R = rotation.toRotationMatrix();

                Eigen::Matrix<double, 3, 4> P;

                P <<
                    R(0, 0), R(0, 1), R(0, 2), translation.x(),
                    R(1, 0), R(1, 1), R(1, 2), translation.y(),
                    R(2, 0), R(2, 1), R(2, 2), translation.z();

                return P;
            }

            inline Eigen::Matrix3d getE() {
                // See wikipedia for Essential matrix for details
                Eigen::Matrix3d tx;
                tx <<
                    0, translation.z(), -translation.y(),
                    -translation.z(), 0, translation.x(),
                    translation.y(), -translation.x(), 0;

                return tx * rotation.toRotationMatrix().transpose();
            }
        };

        struct Observation {
            size_t pointIndex;
            Eigen::Vector2d point;
        };

        template<typename T>
        static inline void computeError(
                const T* const camera_translation,
                const T* const camera_rotation,
                const T* const point3,
                const T* const projectedPoint2,
                T* residuals) {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(camera_translation);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3(point3);
            Eigen::Map<const Eigen::Matrix<T, 2, 1>> p2(projectedPoint2);

            Eigen::Quaternion<T> rotation(
                    camera_rotation[3],
                    camera_rotation[0],
                    camera_rotation[1],
                    camera_rotation[2]);

            // rotation.normalize();

            Eigen::Matrix<T, 3, 1> p3Trans = (rotation * p3) + translation;

            T predicted_x = p3Trans[0] / p3Trans[2];
            T predicted_y = p3Trans[1] / p3Trans[2];

            residuals[0] = predicted_x - projectedPoint2[0];
            residuals[1] = predicted_y - projectedPoint2[1];
        }

        struct CamDepthReprojectionError {
            // (u, v): the position of the observation with respect to the image
            // center point.
            CamDepthReprojectionError(
                    const DepthReconstruction* _self,
                    const Observation* _obs)
                : self(_self), obs(_obs) {}

            template <typename T>
                bool operator()(
                        const T* const camera_translation,
                        const T* const camera_rotation,
                        const T* const depth,
                        T* residuals) const {
                    size_t pointIndex = obs->pointIndex;

                    const Eigen::Vector2d& keypoint = self->keypoints[pointIndex];

                    Eigen::Matrix<T, 3, 1> p;
                    p <<
                        T(keypoint.x()) * depth[0],
                        T(keypoint.y()) * depth[0],
                        depth[0];

                    T projectedPoint2[2];
                    projectedPoint2[0] = T(obs->point[0]);
                    projectedPoint2[1] = T(obs->point[1]);

                    computeError(
                            camera_translation,
                            camera_rotation,
                            p.data(),
                            projectedPoint2,
                            residuals);

                    return true;
                }

            const DepthReconstruction* self;
            const Observation* obs;
        };

        struct CameraReprojectionError {
            CameraReprojectionError(
                    const DepthReconstruction* _self,
                    const Observation* _obs)
                : self(_self), obs(_obs) {}

            template <typename T>
                bool operator()(
                        const T* const camera_translation,
                        const T* const camera_rotation,
                        T* residuals) const {
                    size_t pointIndex = obs->pointIndex;

                    const Eigen::Vector2d& keypoint = self->keypoints[pointIndex];

                    double depth = self->depth[pointIndex];

                    Eigen::Matrix<T, 3, 1> p;
                    p <<
                        T(keypoint.x() * depth),
                        T(keypoint.y() * depth),
                        T(depth);

                    T projectedPoint2[2];
                    projectedPoint2[0] = T(obs->point[0]);
                    projectedPoint2[1] = T(obs->point[1]);

                    computeError(
                            camera_translation,
                            camera_rotation,
                            p.data(),
                            projectedPoint2,
                            residuals);

                    return true;
                }

            const DepthReconstruction* self;
            const Observation* obs;
        };

    public:
        DepthReconstruction();

        void init(
                int numCameras,
                int numPoints);

        inline void setKeypoint(
                size_t pointIndex,
                const Eigen::Vector2d& point) {
            keypoints[pointIndex] = point;
        }

        inline void addObservation(
                size_t cameraIndex,
                size_t pointIndex,
                const Eigen::Vector2d& point) {
            observations[cameraIndex].push_back({pointIndex, point});
        }

        void solve();

        void visualize(
                CImg<float>& depthVis,
                int minInlierCount,
                float outlierPercentile,
                float inlierRangeMultiplier,
                bool printToStdout);

        /**
         * Returns all depth samples resulting from triangulation with
         * each inlier camera.
         *
         * Results are returned as a list containing each keypoint in the main
         * image along with a vector of corresponding depth samples, and a
         * measure of their confidence.
         */
        void getAllDepthSamples(
                vector<tuple<Eigen::Vector2d, vector<tuple<double, double>>>>& depthSamples);

        void getAllDepthSamples(
                size_t cameraI,
                vector<tuple<Eigen::Vector2d, double, double>>& depthSamples);

        inline bool isInlierCamera(
                size_t cameraIndex) {
            return cameraInlierMask[cameraIndex];
        }

        inline size_t getPointCount() const {
            return keypoints.size();
        }

        inline Eigen::Vector3d get3DPoint(
                int pointIndex) const {
            const Eigen::Vector2d& pt = keypoints[pointIndex];

            const double& d = depth[pointIndex];

            return Eigen::Vector3d(pt.x() * d, pt.y() * d, d);
        }

        inline Eigen::Vector2d& getDepthSample(
                int pointIndex,
                double& depthVal,
                size_t& inlierObsCount) {
            depthVal = depth[pointIndex];

            inlierObsCount = inlierCount[pointIndex];

            return keypoints[pointIndex];
        }

        inline Eigen::Matrix3d getEssentialMatrix(
                int cameraIndex) {
            return cameras[cameraIndex].getE();
        }

        bool getPolarFundamentalMatrix(
                int cameraIndex,
                const Eigen::Vector2d& imageCenter,
                double imageSize,
                PolarFundamentalMatrix& polarF);

        inline const vector<Eigen::Vector2d>& getKeypoints() {
            return keypoints;
        }

    private:
        void resetSolutionState();

        void resetInlierMask(
                int cameraIndex);

        /**
         * Estimates the specified camera's fundamental matrix using its
         * observations alone.
         */
        size_t estimateFUsingObs(
                int cameraIndex);

        /**
         * Estimates camera pose from the fundamental matrix.
         *
         * Futher prunes the set of inlier observations to those which fit
         * with the resulting pose.
         *
         * Returns the number of inliers.
         */
        size_t estimatePoseUsingF(
                int cameraIndex);

        /**
         * Uses the specified camera to triangulate depth from inlier observations.
         *
         * Note that only inliers (based on observationInlierMask) for depth values
         * which are uninitialized (0) are modified.
         *
         * Returns the number of new depth samples resulting from triangulation.
         */
        size_t triangulateDepthUsingPose(
                int cameraIndex);

        /**
         * Returns the number of observations for the given camera for which
         * a depth estimate already exists.
         */
        size_t computeDepthCorrespondenceCount(
                int cameraIndex);

        /**
         * Estimates the given camera's parameters by considering that camera's
         * observations as well as any existing depth estimates.
         *
         * Note that this obeys the observationInlierMask and will toggle
         * (logical AND) additional observations as outliers if they don't fit
         * the estimated camera pose.
         *
         * Returns the number of inliers.
         */
        size_t estimatePoseUsingDepth(
                int cameraIndex,
                double inlierThreshold);

        /**
         * Refines estimates of all cameras (for which cameraMask is true) and
         * depths using sparse bundle adjustment.
         *
         * Note that this obeys the observationInlierMask.
         */
        void refineCamerasAndDepth(
                const vector<bool>& cameraMask);

        const double robustLossHuberParam;

        const double inlierThreshold;

        /**
         * Fundamental matrix estimate for each camera relative to the main
         * viewpoint.
         *
         * Note that, since keypoints and observations must already be in
         * normalized device coordinates, these are also essential matrices.
         */
        vector<Eigen::Matrix3d> Fmatrices;

        /*
         * Stores the (x, y) coordates of the observation in the main image
         * as normalized device coordinates (origin is at the center of the
         * image).
         *
         * Thus, the 3D point is actually (x * depth, y * depth, depth).
         */
        vector<Eigen::Vector2d> keypoints;
        vector<size_t> inlierCount;

        /**
         * Stores each observation made from each camera.
         */
        vector<vector<Observation>> observations;
        vector<vector<bool>> observationInlierMask;

        /**
         * A value of 0 for depth indicates an uninitialized value.
         */
        vector<double> depth;

        vector<CameraParam> cameras;
        vector<bool> cameraInlierMask;

        CVFundamentalMatrixEstimator fundMatEstimator;
};

