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

                Matrix3T Kol;
                Kol <<
                    f,    T(0), T(width / 2.0f),
                    T(0), f,    T(height/ 2.0f),
                    T(0), T(0), T(1);

                Matrix3T Kor = Kol;

                Matrix3T Rl;
                Rl =
                    Eigen::AngleAxis<T>(T(0), Vector3T::UnitX()) *
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
                    Kor.inverse().transpose() * Rr.transpose() *
                    hat *
                    Rl * Kol.inverse();

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
                Vector3T m2fu = m2.transpose() * F * star3;

                T num = m2.transpose() * F * m1;
                T denom = ufm1.transpose() * ufm1;
                denom += m2fu.transpose() * m2fu;

                residual[0] = sqrt(num * num / denom);

                /*
                T m2t_F_m1 = m2.transpose() * F * m1;

                Vector3T F_m1 = F * m1;
                Vector3T Ft_m2 = F.transpose() * m2;

                residual[0]  = m2t_F_m1 * m2t_F_m1 / (
                        F_m1[0] * F_m1[0] +
                        F_m1[1] * F_m1[1] +
                        Ft_m2[0] * Ft_m2[0] +
                        Ft_m2[1] * Ft_m2[1]);
                */

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

            CImg<double> warpLeft(left.width(), left.height(), 2);
            CImg<double> warpRight(right.width(), right.height(), 2);

            Eigen::Vector2d pre;
            Eigen::Vector2d post;

            /*
            // FIXME
            cimg_forXY(warp, x, y) {
                pre[0] = (((double) x) / warp.width()) * 2.0f - 1.0f;
                pre[1] = (((double) y) / warp.height()) * 2.0f - 1.0f;

                transformPoint(
                        distortion.data(),
                        &(transforms[camera * 9]),
                        pre.data(),
                        post.data());

                warp(x, y, 0) = (post[0] + 1.0f) / 2.0f * warp.width();
                warp(x, y, 1) = (post[1] + 1.0f) / 2.0f * warp.height();
            }

            warped = original.get_warp(warp, false, 2, 0);
            */
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
