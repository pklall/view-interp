#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "common.h"

#include "reconstruct.h"

/**
 * Stereo rectification based on "Quasi-Euclidean Uncalibrated Epipolar
 * Rectification" by Fusiello and Isara (2008).
 */
class Rectification {
    public:
        typedef Eigen::Matrix<double, 6, 1> TransformParams;

    private:
        struct TransformErrorFunction {
            Eigen::Vector2f left, right;

            int width, height;

            template <typename T>
            bool operator()(
                    const T* const params,
                    T* residual) const {
                typedef Eigen::Matrix<T, 2, 1> Vector2T;
                typedef Eigen::Matrix<T, 3, 1> Vector3T;
                typedef Eigen::Matrix<T, 3, 3> Matrix3T;

                T yl = params[0];
                T zl = params[1];

                T xr = params[2];
                T yr = params[3];
                T zr = params[4];

                // 3^a(6)*(w + h)
                T f = exp(T(log(3.0)) * params[5]) * T(width + height);

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

                // Compute Sampson residual

                Vector3T m1;
                m1 << T(left[0]), T(left[1]), T(1);

                Vector3T m2;
                m2 << T(right[0]), T(right[1]), T(1);

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

    public:
        Rectification(
                const ChainFeatureMatcher* _features,
                Eigen::Vector2i _imageSize);

        void print(
                ostream& result) const;

        void solve(
                int numRestarts);

        template<class T>
        void warp(
                const CImg<T>& left,
                const CImg<T>& right,
                CImg<T>& warpedLeft,
                CImg<T>& warpedRight) const {
            Eigen::Matrix3f ml, mr;

            paramsToMat(transform, ml, mr);

            Eigen::Matrix<float, 3, 4> cornersLeft;
            cornersLeft <<
                0.0, 0.0           , left.width() , left.width(),
                0.0, left.height() , left.height(), 0.0,
                1.0, 1.0           , 1.0          , 1.0;

            Eigen::Matrix<float, 3, 4> cornersRight;
            cornersRight <<
                0.0, 0.0           , right.width() , right.width(),
                0.0, right.height(), right.height(), 0.0,
                1.0, 1.0           , 1.0           , 1.0;

            cornersLeft = ml * cornersLeft;
            cornersRight = mr * cornersRight;

            cornersLeft.row(0) = cornersLeft.row(0).cwiseQuotient(cornersLeft.row(2));
            cornersLeft.row(1) = cornersLeft.row(1).cwiseQuotient(cornersLeft.row(2));
            cornersRight.row(0) = cornersRight.row(0).cwiseQuotient(cornersRight.row(2));
            cornersRight.row(1) = cornersRight.row(1).cwiseQuotient(cornersRight.row(2));

            Eigen::Vector2f minLeft(
                    cornersLeft.row(0).minCoeff(),
                    cornersLeft.row(1).minCoeff());
            Eigen::Vector2f maxLeft(
                    cornersLeft.row(0).maxCoeff(),
                    cornersLeft.row(1).maxCoeff());
            Eigen::Vector2f minRight(
                    cornersRight.row(0).minCoeff(),
                    cornersRight.row(1).minCoeff());
            Eigen::Vector2f maxRight(
                    cornersRight.row(0).maxCoeff(),
                    cornersRight.row(1).maxCoeff());

            float minY = min(minLeft[1], minRight[1]);
            float maxY = min(maxLeft[1], maxRight[1]);

            float targetHeight = min(warpedLeft.height(), warpedRight.height());

            auto verticalTransform =
                Eigen::AlignedScaling2f(1.0, targetHeight / (maxY - minY)) *
                Eigen::Translation2f(0, -minLeft[1]);

            ml = verticalTransform *
                Eigen::AlignedScaling2f(warpedLeft.width() / (maxLeft[0] - minLeft[0]), 1.0) *
                Eigen::Translation2f(-minLeft[0], 0) *
                ml;

            mr = verticalTransform *
                Eigen::AlignedScaling2f(warpedRight.width() / (maxRight[0] - minRight[0]), 1.0) *
                Eigen::Translation2f(-minRight[0], 0) *
                mr;

            Eigen::Matrix3f mlInv = ml.inverse();

            Eigen::Matrix3f mrInv = mr.inverse();

            Eigen::Vector3f v;

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
                Eigen::Matrix3f& ml,
                Eigen::Matrix3f& mr) const;

        unique_ptr<ceres::Problem> createProblem(
                function<void(int, Eigen::Vector2f&, Eigen::Vector2f&)> pairGen,
                int numPairs,
                bool robustify,
                TransformParams& params) const;

        const ChainFeatureMatcher* features;

        Eigen::Vector2i imageSize;

        double residual;

        TransformParams transform;
};
