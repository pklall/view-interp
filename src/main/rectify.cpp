#include "rectify.h"

void Rectification::init(
        Eigen::Vector2i _imageSize,
        const ChainFeatureMatcher* features) {
    imageSize = _imageSize;

    const vector<vector<tuple<int, Eigen::Vector2f>>>& observations =
        features->getObservations();

    matches.clear();

    matches.resize(observations[0].size());

    for (const auto& obs : observations[0]) {
        get<0>(matches[get<0>(obs)]) = get<1>(obs);
    }

    // TODO Handle case in which not all matches from image 0's match list
    //      have a corresponding element for image 1.
    for (const auto& obs : observations[1]) {
        int ptIndex = get<0>(obs);

        if (ptIndex < matches.size()) {
            get<1>(matches[get<0>(obs)]) = get<1>(obs);
        }
    }

    initErrorTerms();

    residual = std::numeric_limits<double>::max();

    transform.fill(0.0);
}

void Rectification::print(
        ostream& result) const {
    Eigen::Matrix3d ml, mr;

    paramsToMat(transform, ml, mr);

    result << "Raw = " << endl << transform << endl;

    result << "M_left = " << endl << ml << endl;

    result << "M_right = " << endl << mr << endl;
}

void Rectification::solve(
        int numRestarts) {
    residual = std::numeric_limits<double>::max();

    const int matchesPerRestart = 10;

    initErrorTerms();

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;

    ceres::HuberLoss* robustLoss = new ceres::HuberLoss(0.01);

    ceres::Problem::Options pOptions;
    pOptions.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    pOptions.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    // Store a map from cost to index of error term with that cost
    vector<tuple<double, int>> costs(errorTerms.size());

    double curResidual;
    TransformParams curTransform;

    int bestNumInliers = 0;

    for (int i = 0; i < numRestarts; i++) {
        ceres::Problem localProblem(pOptions);

        // Clear the initial parameters
        curTransform.fill(0.0);

        // Fit an initial transform to a selected few points with least-squares
        for (int m = 0; m < matchesPerRestart; m++) {
            int residualIndex = (i * (matchesPerRestart / 5) + m) % errorTerms.size();

            localProblem.AddResidualBlock(
                    errorTerms[residualIndex].get(),
                    nullptr,
                    curTransform.data());
        }

        ceres::Solve(options, &localProblem, &summary);

        // Evaluate the current model against all error terms to detect
        // outliers
        computeEpipolarCosts(curTransform, costs);

        std::sort(costs.begin(), costs.end());

        ceres::Problem globalProblem(pOptions);

        int numInliers = 0;

        const double inlierThresh = 0.05;

        for (int t = 0; t < costs.size(); t++) {
            if (get<0>(costs[t]) > inlierThresh && t > 30) {
                break;
            }

            globalProblem.AddResidualBlock(
                    errorTerms[get<1>(costs[t])].get(),
                    robustLoss,
                    curTransform.data());

            numInliers++;
        }

        ceres::Solve(options, &globalProblem, &summary);

        // Compute the number of inliers after refinement
        computeEpipolarCosts(curTransform, costs);

        numInliers = 0;
        for (const auto& cost : costs) {
            if (get<0>(cost) <= inlierThresh) {
                numInliers++;
            }
        }

        curResidual = summary.final_cost;

        if (bestNumInliers < numInliers ||
                (bestNumInliers == numInliers &&
                 curResidual < residual)) {
            residual = curResidual;
            transform = curTransform;
            bestNumInliers = numInliers;
        }
    }
}

void Rectification::estimateDisparityRange(
        int outputWidth,
        int outputHeight,
        float& minDisp,
        float& maxDisp) const {
    Eigen::Matrix3d ml, mr;

    paramsToMat(transform,
            outputWidth, outputHeight,
            ml, mr);

    vector<tuple<double, int>> costs(errorTerms.size());
    
    computeCosts(transform, costs);

    int inlierCount = (costs.size()) * 0.75;

    std::partial_sort(
            &(costs[0]),
            &(costs[inlierCount]),
            &(costs[costs.size()]));

    minDisp = std::numeric_limits<float>::max();
    maxDisp = std::numeric_limits<float>::min();

    // Only consider inliers
    for (const auto& inlier : costs) {
        const auto& match = matches[get<1>(inlier)];

        const Eigen::Vector2f& left = get<0>(match);
        const Eigen::Vector2f& right = get<1>(match);

        Eigen::Vector3d leftR = ml * Eigen::Vector3d(left[0], left[1], 1.0);
        Eigen::Vector3d rightR = mr * Eigen::Vector3d(right[0], right[1], 1.0);

        double disp = rightR[0] / rightR[2] - leftR[0] / leftR[2];
        
        minDisp = min((double) minDisp, disp);
        maxDisp = max((double) maxDisp, disp);
    }
}

void Rectification::paramsToMat(
        const TransformParams& params,
        Eigen::Matrix3d& ml,
        Eigen::Matrix3d& mr) const {
    float yl = params[0];
    float zl = params[1];

    float xr = params[2];
    float yr = params[3];
    float zr = params[4];

    float f = exp(log(3.0) * params[5]) * (imageSize[0] + imageSize[1]);

    Eigen::Matrix3d K;
    K <<
        f, 0, imageSize[0] / 2.0f,
        0, f, imageSize[1] / 2.0f,
        0, 0, 1;

    Eigen::Matrix3d Rl;
    Rl =
        Eigen::AngleAxisd(0,  Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(zl, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(yl, Eigen::Vector3d::UnitY());

    Eigen::Matrix3d Rr;
    Rr =
        Eigen::AngleAxisd(xr, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(zr, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(yr, Eigen::Vector3d::UnitY());

    auto Kinverse = K.inverse();

    ml = K * Rl * Kinverse;
    mr = K * Rr * Kinverse;
}

void Rectification::paramsToMat(
        const TransformParams& params,
        int outputWidth,
        int outputHeight,
        Eigen::Matrix3d& ml,
        Eigen::Matrix3d& mr) const {
    paramsToMat(transform, ml, mr);

    Eigen::Matrix<double, 3, 4> cornersLeft;
    cornersLeft <<
        0.0, 0.0         , imageSize[0], imageSize[0],
        0.0, imageSize[1], imageSize[1], 0.0,
        1.0, 1.0         , 1.0         , 1.0;

    Eigen::Matrix<double, 3, 4> cornersRight;
    cornersRight <<
        0.0, 0.0         , imageSize[0], imageSize[0],
        0.0, imageSize[1], imageSize[1], 0.0,
        1.0, 1.0         , 1.0         , 1.0;

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

    // There's no reason to include parts of the left or right image
    // which are entirely above or below the other when transformed
    // since depth cannot be computed.
    // So, the following give a bound on the relevant vertical range
    // of the transformed images.
    float maxminY = max(minLeft[1], minRight[1]);
    float minmaxY = min(maxLeft[1], maxRight[1]);

    float targetHeight = outputHeight;

    auto verticalTransform =
        Eigen::AlignedScaling2d(1.0, targetHeight / (minmaxY - maxminY)) *
        Eigen::Translation2d(0, -maxminY);

    ml = verticalTransform *
        Eigen::AlignedScaling2d(
                outputWidth / (maxLeft[0] - minLeft[0]),
                1.0) *
        Eigen::Translation2d(-minLeft[0], 0) *
        ml;

    mr = verticalTransform *
        Eigen::AlignedScaling2d(
                outputWidth / (maxRight[0] - minRight[0]),
                1.0) *
        Eigen::Translation2d(-minRight[0], 0) *
        mr;
}

void Rectification::computeCosts(
        const TransformParams& transform,
        vector<tuple<double, int>>& costs) const {
    costs.resize(errorTerms.size());

    for (int t = 0; t < errorTerms.size(); t++) {
        get<1>(costs[t]) = t;

        double const * const paramBlock[1] = {transform.data()};

        errorTerms[t]->Evaluate(
                paramBlock,
                &get<0>(costs[t]),
                nullptr);
    }
}

void Rectification::computeEpipolarCosts(
        const TransformParams& transform,
        vector<tuple<double, int>>& costs) const {
    costs.resize(matches.size());

    Eigen::Matrix3d ml, mr;

    paramsToMat(transform,
            1.0, 1.0,
            ml, mr);

    for (int i = 0; i < matches.size(); i++) {
        const auto& match = matches[i];

        const Eigen::Vector2f& left = get<0>(match);
        const Eigen::Vector2f& right = get<1>(match);

        Eigen::Vector3d leftR = ml * Eigen::Vector3d(left[0], left[1], 1.0);
        Eigen::Vector3d rightR = mr * Eigen::Vector3d(right[0], right[1], 1.0);

        double error = abs(rightR[1] / rightR[2] - leftR[1] / leftR[2]);

        costs[i] = make_tuple(error, i);
    }
}

void Rectification::initErrorTerms() {
    errorTerms.clear();

    errorTerms.reserve(matches.size());

    for (int i = 0; i < matches.size(); i++) {
        TransformCostFunction* func = new TransformCostFunction();

        func->self = this;
        func->matchIndex = i;

        errorTerms.push_back(
                unique_ptr<ADTransformCostFunc>(
                    new ADTransformCostFunc(func)));
    }
}
