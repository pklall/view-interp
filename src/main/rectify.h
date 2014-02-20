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
                T totalY;

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
         * Enforces unit determinant of a 3x3 matrix with a single residual.
         */
        struct UnitDetPrior {
            template<typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;

                Matrix3TConst t(transform);

                residual[0] = t.determinant() - T(1);

                return true;
            }
        };

        /**
         * Creates a single residual based on the given parameter.
         */
        struct SmallPrior {
            template<typename T>
            bool operator()(
                    const T* const param,
                    T* residual) const {
                residual[0] = param;

                return true;
            }
        };

        /**
         * Enforces the width of the transformed unit-square to be unit using 2
         * residuals.
         */
        struct TransformUnitWidthPrior {
            template<typename T>
            bool operator()(
                    const T* const transform,
                    T* residual) const {
                typedef Eigen::Map<const Eigen::Matrix<T, 3, 3>> Matrix3TConst;
                typedef Eigen::Matrix<T, 3, 1> Vector3T;

                Matrix3TConst t(transform);

                Vector3T topLeft(T(0), T(0), T(1));
                Vector3T topRight(T(1), T(0), T(1));

                Vector3T botLeft(T(0), T(1), T(1));
                Vector3T botRight(T(1), T(1), T(1));

                topLeft = t * topLeft;
                topRight = t * topRight;

                botLeft = t * botLeft;
                botRight = t * botRight;
                
                residual[0] = (topRight[0] - topLeft[0]) - T(1.0f);
                residual[1] = (botRight[0] - botLeft[0]) - T(1.0f);

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

            int numPoints = features->getNumPoints();

            // Set initial transform values
            distortion.fill(0.0f);

            for (int i = 0; i < NCameras; i++) {
                typedef Eigen::Map<Eigen::Matrix3d> Matrix3DMap;
                Matrix3DMap t(&(transforms[i * 9]));
                t = Eigen::Matrix3d::Identity();
            }

            ceres::Problem problem;

            ceres::LossFunction* squareLoss = NULL;

            ceres::LossFunction* robustLoss = new ceres::HuberLoss(3.0);

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

            ceres::Solver::Options options;
            // options.linear_solver_type = ceres::LE
            options.max_num_iterations = 10000;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cout << summary.FullReport() << endl;
        }

};
