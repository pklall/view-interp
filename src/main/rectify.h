#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "common.h"

#include "reconstruct.h"

template<typename T>
inline void radialUndistort(
        const Eigen::Matrix<T, 2, 1>& original,
        const T* distortion,
        Eigen::Matrix<T, 2, 1>& result) {
    // See Eq. 5 of Precise Radial Un-distortion of Images, Mallon &
    // Whelan, 2004.
    T rd2 =
        T(original[0]) * T(original[0]) +
        T(original[1]) * T(original[1]);
    T rd4 = rd2 * rd2;
    T rd6 = rd4 * rd2;
    T rd8 = rd6 * rd2;

    T numCoeff = 
        T(distortion[0]) * rd2 +
        T(distortion[1]) * rd4 +
        T(distortion[2]) * rd6 +
        T(distortion[3]) * rd8;

    T denom = T(1.0) +
        T(4.0) * T(distortion[4]) * rd2 +
        T(6.0) * T(distortion[5]) * rd4;

    result[0] = T(original[0]) - T(original[0]) * numCoeff / denom;
    result[1] = T(original[1]) - T(original[1]) * numCoeff / denom;
}

/**
 * Stereo rectification based on "Quasi-Euclidean Uncalibrated Epipolar
 * Rectification" by Fusiello and Isara (2008).
 */
class Rectification {
    public:
        /**
         * Intrinsics include the focal length term of Fusiello and Isara
         * followed by 6 parameters for the radial distortion model
         * of Mallon & Whelan.
         */
        typedef Eigen::Matrix<double, 6, 1> TransformParams;

    private:
        /**
         * Cost function based on "Quasi-Euclidean Uncalibrated Epipolar
         * Rectification" by Fusiello and Isara (2008).
         */
        struct TransformCostFunction {
            Rectification* self;

            int matchIndex;

            template <typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                typedef Eigen::Matrix<T, 2, 1> Vector2T;
                typedef Eigen::Matrix<T, 3, 1> Vector3T;
                typedef Eigen::Matrix<T, 3, 3> Matrix3T;

                int width = self->imageSize[0];
                int height = self->imageSize[1];

                const Eigen::Vector2f& left = get<0>(self->matches[matchIndex]);
                const Eigen::Vector2f& right = get<1>(self->matches[matchIndex]);

                T yl = transform[0];
                T zl = transform[1];

                T xr = transform[2];
                T yr = transform[3];
                T zr = transform[4];

                // 3^a(6)*(w + h)
                T f = exp(T(log(3.0)) * transform[5]) * T(width + height);

                Matrix3T K;
                K <<
                    f,    T(0), T(width / 2.0f),
                    T(0), f,    T(height/ 2.0f),
                    T(0), T(0), T(1);

                Matrix3T Kinv = K.inverse();

                Matrix3T Rl;
                Rl =
                    // Eigen::AngleAxis<T>(T(0), Vector3T::UnitX()) *
                    Eigen::AngleAxis<T>(zl,   Vector3T::UnitZ()) *
                    Eigen::AngleAxis<T>(yl,   Vector3T::UnitY());

                Matrix3T Rr;
                Rr =
                    Eigen::AngleAxis<T>(xr, Vector3T::UnitX()) *
                    Eigen::AngleAxis<T>(zr, Vector3T::UnitZ()) *
                    Eigen::AngleAxis<T>(yr, Vector3T::UnitY());

                Matrix3T hat;
                hat <<
                    T(0), T(0), T(0),
                    T(0), T(0), T(-1),
                    T(0), T(1), T(0);

                Matrix3T F =
                    Kinv.transpose() * Rr.transpose() *
                    hat *
                    Rl * Kinv;

                // Invert radial distortion
                Vector2T leftU;
                Vector2T rightU;

                /*
                radialUndistort(
                        Vector2T(T(left[0]), T(left[1])),
                        &(transform[6]),
                        leftU);

                radialUndistort(
                        Vector2T(T(right[0]), T(right[1])),
                        &(transform[6]),
                        rightU);
                */
                leftU = Vector2T(T(left[0]), T(left[1]));
                rightU = Vector2T(T(right[0]), T(right[1]));


                // Compute Sampson residual
                Vector3T m1;
                m1 << leftU[0], leftU[1], T(1);

                Vector3T m2;
                m2 << rightU[0], rightU[1], T(1);

                Matrix3T star3;
                star3 <<
                    T(0), T(-1), T(0),
                    T(1), T(0),  T(0),
                    T(0), T(1),  T(0);

                Vector3T ufm1 = star3 * F * m1;
                Vector3T m2fu = (m2.transpose() * F * star3).transpose();

                T num = m2.transpose() * F * m1;
                T denom = ufm1.transpose() * ufm1;
                denom += m2fu.transpose() * m2fu;

                residual[0] = sqrt(num * num / denom);

                return true;
            }
        };

        typedef ceres::AutoDiffCostFunction<TransformCostFunction, 1, 6>
            ADTransformCostFunc;

    public:
        void init(
                Eigen::Vector2i _imageSize,
                const ChainFeatureMatcher* _features);

        void print(
                ostream& result) const;

        void solve(
                int numRestarts);

        void estimateDisparityRange(
                int outputWidth,
                int outputHeight,
                float& minDisp,
                float& maxDisp) const;

        template<class T>
        void warp(
                const CImg<T>& left,
                const CImg<T>& right,
                CImg<T>& warpedLeft,
                CImg<T>& warpedRight) const {
            assert(left.width() == imageSize[0]);
            assert(left.height() == imageSize[1]);
            assert(right.width() == imageSize[0]);
            assert(right.height() == imageSize[1]);

            Eigen::Matrix3d ml, mr;

            int width = min(warpedLeft.width(), warpedRight.width());
            int height = min(warpedLeft.height(), warpedRight.height());

            paramsToMat(transform,
                    width, height, ml, mr);

            Eigen::Matrix3d mlInv = ml.inverse();

            Eigen::Matrix3d mrInv = mr.inverse();

            Eigen::Vector3d v;

            cimg_forXY(warpedLeft, x, y) {
                v[0] = x;
                v[1] = y;
                v[2] = 1.0f;

                v = mlInv * v;

                cimg_forC(warpedLeft, c) {
                    warpedLeft(x, y, 0, c) = left.linear_atXY(
                            v[0] / v[2], v[1] / v[2], 0, c, T(0));
                }
            }

            cimg_forXY(warpedRight, x, y) {
                v[0] = x;
                v[1] = y;
                v[2] = 1.0f;

                v = mrInv * v;

                cimg_forC(warpedRight, c) {
                    warpedRight(x, y, 0, c) = right.linear_atXY(
                            v[0] / v[2], v[1] / v[2], 0, c, T(0));
                }
            }
        }

    private:
        void paramsToMat(
                const TransformParams& params,
                Eigen::Matrix3d& ml,
                Eigen::Matrix3d& mr) const;

        void paramsToMat(
                const TransformParams& params,
                int outputWidth,
                int outputHeight,
                Eigen::Matrix3d& ml,
                Eigen::Matrix3d& mr) const;

        /**
         * Evalutes each error term on the given transform returning tuples
         * of the form (cost, error term #).
         */
        void computeCosts(
                const TransformParams& transform,
                vector<tuple<double, int>>& costs) const;

        /**
         * Computes the vertical difference between matched points transformed
         * by the given parameter.  This difference is normalized assuming
         * that the output image has a height of 1.
         *
         * Tuples of the form (cost, match #) are returned.
         */
        void computeEpipolarCosts(
                const TransformParams& transform,
                vector<tuple<double, int>>& costs) const;

        void initErrorTerms();

        Eigen::Vector2i imageSize;

        vector<tuple<Eigen::Vector2f, Eigen::Vector2f>> matches;

        // Contains pointers to each error term in the current problem
        vector<unique_ptr<ADTransformCostFunc>> errorTerms;

        // The residual associated with the best transform found so far
        double residual;

        // The best transform found so far
        TransformParams transform;

        int numInliers;
};

