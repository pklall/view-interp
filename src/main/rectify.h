#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "common.h"

#include "reconstruct.h"

/**
 * Stereo rectification based on
 * "Quasi-Euclidean Uncalibrated Epipolar Rectification" by Fusiello and Isara (2008).
 */
class Rectification {
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
                    Eigen::AngleAxis<T>(yl, Vector3T::UnitY()) *
                    Eigen::AngleAxis<T>(zl, Vector3T::UnitZ()) *
                    Eigen::AngleAxis<T>(T(0), Vector3T::UnitX());

                Matrix3T Rr;
                Rr =
                    Eigen::AngleAxis<T>(yr, Vector3T::UnitY()) *
                    Eigen::AngleAxis<T>(zr, Vector3T::UnitZ()) *
                    Eigen::AngleAxis<T>(xr, Vector3T::UnitX());

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

                T m2t_F_m1 = m2.transpose() * F * m1;

                Vector3T F_m1 = F * m1;
                Vector3T Ft_m2 = F.transpose() * m2;

                residual[0]  = m2t_F_m1 * m2t_F_m1 / (
                        F_m1[0] * F_m1[0] +
                        F_m1[1] * F_m1[1] +
                        Ft_m2[0] * Ft_m2[0] +
                        Ft_m2[1] * Ft_m2[1]);

                return true;
            }
        };

        struct Prior {
            template<typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                residual[0] = T(0);
                return true;
            }
        };

        const ChainFeatureMatcher* features;

        array<double, 6> transformParams;

        Eigen::Vector2i imageSize;

    public:
        Rectification(
                const ChainFeatureMatcher* _features,
                Eigen::Vector2i _imageSize) :
                features(_features), 
                imageSize(_imageSize) {
        }

        void solve() {
            const auto& matches = features->getObservations();

            // Set initial transform values
            transformParams.fill(0.0f);

            ceres::Problem problem;

            // FIXME
            ceres::LossFunction* robustLoss = NULL;//new ceres::HuberLoss(3.0);

            // Allocate an error function for each matched point
            vector<TransformErrorFunction*> errorFuncs(matches[0].size());

            for (TransformErrorFunction*& func : errorFuncs) {
                func = new TransformErrorFunction();
                func->width = imageSize[0];
                func->height = imageSize[1];
            }

            for (const auto& match : matches[0]) {
                errorFuncs[get<0>(match)]->left = get<1>(match);
            }

            for (const auto& match : matches[1]) {
                int ptIndex = get<0>(match);

                if (ptIndex >= errorFuncs.size()) {
                    continue;
                } else {
                    errorFuncs[ptIndex]->right = get<1>(match);

                    typedef ceres::AutoDiffCostFunction<
                        TransformErrorFunction, 1, 6>
                        AutoDiffErrorFunc;

                    ceres::CostFunction* costFunction =
                        new AutoDiffErrorFunc(errorFuncs[ptIndex]);

                    problem.AddResidualBlock(
                            costFunction,
                            robustLoss,
                            transformParams.data());
                }
            }

            ceres::Solver::Options options;
            options.max_num_iterations = 10000;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cout << summary.FullReport() << endl;
        }

        void print(
                ostream& result) const {
            // FIXME
        }

        template<class T>
        void warp(
                const CImg<T>& original,
                CImg<T>& warped) {
            CImg<double> warp(original.width(), original.height(), 2);

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
};
