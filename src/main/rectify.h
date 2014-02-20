#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "common.h"

#include "reconstruct.h"

template<int NCameras>
class ChainRectification {
    private:
        /**
         * Transforms 2D-point `original` by first performing inverse radial
         * distortion via the method of Mallon & Whelan (2004) with 6 parameters
         * from `distortion` followed by multiplication by the 3x3 matrix
         * `transform`.
         */
        template<typename T, typename T2>
        static inline void transformPoint(
                const T* const distortion,
                const T* const transform,
                const T2* const original,
                T* transformed) {
            typedef Eigen::Matrix<T, 3, 1> Vector3T;
            typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;

            Matrix3TConst trans(transform);

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

            T denom = T(1) +
                T(4) * T(distortion[4]) * rd2 +
                T(6) * T(distortion[5]) * rd4;

            T pointUX = T(original[0]) - T(original[0]) * numCoeff / denom;
            T pointUY = T(original[1]) - T(original[1]) * numCoeff / denom;

            // T pointUX = T(original[0]);
            // T pointUY = T(original[1]);

            Vector3T o(pointUX, pointUY, T(1));

            Vector3T result = trans * o;

            transformed[0] = result[0] / result[2];
            transformed[1] = result[1] / result[2];
        }

        struct TransformErrorFunction {
            /**
             * The set of observations of the form: (camera index, point)
             */
            array<tuple<bool, Eigen::Vector2f>, NCameras> points;

            template <typename T>
            bool operator()(
                    const T* const distortion,
                    const T* const transforms,
                    T* residuals) const {
                typedef Eigen::Matrix<T, 2, 1> Vector2T;

                array<Vector2T, NCameras> transformed;
                T totalY = T(0.0f);

                int numPoints = 0;

                for (int i = 0; i < NCameras; i++) {
                    if (get<0>(points[i])) {
                        const Eigen::Vector2f& observation = get<1>(points[i]);

                        transformPoint(
                                distortion,
                                &(transforms[9 * i]),
                                observation.data(),
                                transformed[i].data());

                        totalY += transformed[i][1];

                        numPoints++;
                    }
                }

                if (numPoints == 0) {
                    for (int i = 0; i < NCameras; i++) {
                        residuals[i] = T(0);
                    }
                }

                totalY /= T(numPoints);
                
                for (int i = 0; i < NCameras; i++) {
                    if (get<0>(points[i])) {
                        residuals[i] = transformed[i][1] - totalY;
                    } else {
                        residuals[i] = T(0);
                    }
                }

                return true;
            }
        };

        /**
         * Enforces unit determinant of a sequence of 3x3 matrices, each with
         * its own residual.
         */
        template<int NumMatrices>
        struct UnitDetPrior {
            template<typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;

                for (int i = 0; i < NumMatrices; i++) {
                    Matrix3TConst t(transform + 9 * i);

                    residual[i] = t.determinant() - T(1);
                }

                return true;
            }
        };

        /**
         * Creates a residual based on the given parameter.
         */
        template<int size>
        struct SmallPrior {
            template<typename T>
            bool operator()(
                    const T* const param,
                    T* residual) const {
                for (int i = 0; i < size; i++) {
                    residual[i] = param[i];
                }

                return true;
            }
        };

        /**
         * Enforces the transformed unit-square to be "sane" using 6 residuals
         * per matrix.
         */
        template<int NumMatrices>
        struct TransformPrior {
            template<typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;
                typedef Eigen::Matrix<T, 3, 1> Vector3T;

                for (int i = 0; i < NumMatrices; i++) {
                    Matrix3TConst t(&(transform[9 * i]));

                    Vector3T topLeft(T(0), T(0), T(1));
                    Vector3T topRight(T(1), T(0), T(1));

                    Vector3T botLeft(T(0), T(1), T(1));
                    Vector3T botRight(T(1), T(1), T(1));

                    topLeft = t * topLeft;
                    topRight = t * topRight;

                    botLeft = t * botLeft;
                    botRight = t * botRight;

                    residual[6 * i + 0] = (topRight[0] - topLeft[0]) - T(1.0f);
                    residual[6 * i + 1] = (botRight[0] - botLeft[0]) - T(1.0f);
                    residual[6 * i + 2] = (topRight[1] - botRight[1]) - T(1.0f);
                    residual[6 * i + 3] = (topLeft[1] - botLeft[1]) - T(1.0f);

                    residual[6 * i + 4] = topLeft[0];
                }

                return true;
            }
        };

        const ChainFeatureMatcher* features;

        array<double, 9> distortion;
        array<double, NCameras * 9> transforms;

    public:
        ChainRectification(
                const ChainFeatureMatcher* _features) :
                features(_features) {
            assert(features->getObservations().size() == NCameras);
        }

        void solve() {
            const auto& matches = features->getObservations();

            // Set initial transform values
            distortion.fill(0.0f);

            for (int i = 0; i < NCameras; i++) {
                typedef Eigen::Map<Eigen::Matrix3d> Matrix3DMap;
                Matrix3DMap t(&(transforms[i * 9]));
                t = Eigen::Matrix3d::Identity();
            }

            ceres::Problem problem;

            ceres::LossFunction* robustLoss = NULL;//new ceres::HuberLoss(3.0);

            // Allocate an error function for each point
            vector<TransformErrorFunction*> errorFuncs(features->getNumPoints());

            for (TransformErrorFunction*& func : errorFuncs) {
                func = new TransformErrorFunction();

                func->points.fill(make_tuple(false, Eigen::Vector2f::Zero()));
            }

            for (int camI = 0; camI < matches.size(); camI++) {
                const vector<tuple<int, float, float>>& camMatches = matches[camI];

                for (const auto& match : camMatches) {
                    int ptIndex = get<0>(match);
                    float x = get<1>(match);
                    float y = get<2>(match);

                    errorFuncs[ptIndex]->points[camI] = 
                            make_tuple(true, Eigen::Vector2f(x, y));
                }
            }

            for (const auto& func : errorFuncs) {
                typedef ceres::AutoDiffCostFunction<
                    TransformErrorFunction, NCameras, 6, NCameras * 9>
                    AutoDiffErrorFunc;

                ceres::CostFunction* costFunction = new AutoDiffErrorFunc(func);

                problem.AddResidualBlock(
                        costFunction,
                        robustLoss,
                        distortion.data(),
                        transforms.data());
            }

            /*
            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<UnitDetPrior<NCameras>, NCameras,
                        NCameras * 9>(
                            new UnitDetPrior<NCameras>()),
                    NULL,
                    transforms.data());
            */

            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<TransformPrior<NCameras>,
                        NCameras * 6,
                        NCameras * 9>(
                            new TransformPrior<NCameras>()),
                    NULL,
                    transforms.data());

            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<SmallPrior<6>, 6, 6>(
                        new SmallPrior<6>()),
                    NULL,
                    distortion.data());

            ceres::Solver::Options options;
            options.max_num_iterations = 10000;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cout << summary.FullReport() << endl;
        }

        void print(
                ostream& result) const {
            result << "Distortion:" << endl;
            
            for (const double& d : distortion) {
                result << "\t" << d << endl;
            }

            result << endl;

            result << "Matrices:" << endl;

            for (int i = 0; i < NCameras; i++) {
                typedef Eigen::Map<const Eigen::Matrix3d> Matrix3DMap;

                Matrix3DMap t(&(transforms[i * 9]));

                for (int y = 0; y < 3; y++) {
                    for (int x = 0; x < 3; x++) {
                        result << t(x, y) << " ";
                    }

                    result << endl;
                }

                result << endl;
            }
        }

        template<class T>
        void warp(
                int camera,
                const CImg<T>& original,
                CImg<T>& warped) {
            CImg<double> warp(original.width(), original.height(), 2);

            Eigen::Vector2d pre;
            Eigen::Vector2d post;

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
        }
};
