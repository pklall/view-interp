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
    Eigen::Matrix3f ml, mr;

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

    // Use the top 50% as inliers
    int inlierCount = (costs.size()) * 0.75;

    double curResidual;
    TransformParams curTransform;

    for (int i = 0; i < numRestarts; i++) {
        ceres::Problem localProblem(pOptions);

        // Clear the initial parameters
        curTransform.fill(0.0);

        // Fit an initial transform to a selected few points with least-squares
        for (int m = 0; m < matchesPerRestart; m++) {
            int residualIndex = (i * matchesPerRestart + m) % errorTerms.size();

            localProblem.AddResidualBlock(
                    errorTerms[residualIndex].get(),
                    nullptr,
                    curTransform.data());
        }

        ceres::Solve(options, &localProblem, &summary);

        // Evaluate the current model against all error terms to detect
        // outliers
        computeCosts(curTransform, costs);

        std::partial_sort(
                &(costs[0]),
                &(costs[inlierCount]),
                &(costs[costs.size()]));

        ceres::Problem globalProblem(pOptions);

        for (int t = 0; t < inlierCount; t++) {
            globalProblem.AddResidualBlock(
                    errorTerms[get<1>(costs[t])].get(),
                    robustLoss,
                    curTransform.data());
        }

        ceres::Solve(options, &globalProblem, &summary);

        curResidual = summary.final_cost;

        if (curResidual >= 0 && curResidual < residual) {
            residual = curResidual;

            transform = curTransform;

            printf("Found improved transform with residual: %f\n", curResidual);

            print(std::cout);
        } else {
            printf("Found inferior transform with residual: %f\n", curResidual);
        }
    }
}

void Rectification::estimateDisparityRange(
        int inputWidth,
        int inputHeight,
        int outputWidth,
        int outputHeight,
        float& minDisp,
        float& maxDisp) const {
    Eigen::Matrix3f ml, mr;

    paramsToMat(transform,
            inputWidth, inputHeight,
            outputWidth, outputHeight,
            ml, mr);

    vector<tuple<double, int>> costs(errorTerms.size());
    
    computeCosts(transform, costs);

    int inlierCount = (costs.size()) * 0.75;

    std::partial_sort(
            &(costs[0]),
            &(costs[inlierCount]),
            &(costs[costs.size()]));

    // Only consider inliers
    for (const auto& inlier : costs) {
        const auto& match = matches[get<1>(inlier)];

        const Eigen::Vector2f& left = get<0>(match);
        const Eigen::Vector2f& right = get<1>(match);

        Eigen::Vector3f leftR = ml * Eigen::Vector3f(left[0], left[1], 1.0);
        Eigen::Vector3f rightR = mr * Eigen::Vector3f(right[0], right[1], 1.0);

        float disp = rightR[0] / rightR[2] - leftR[0] / leftR[2];
    }
}

void Rectification::paramsToMat(
        const TransformParams& params,
        Eigen::Matrix3f& ml,
        Eigen::Matrix3f& mr) const {
    float yl = params[0];
    float zl = params[1];

    float xr = params[2];
    float yr = params[3];
    float zr = params[4];

    float f = exp(log(3.0) * params[5]) * (imageSize[0] + imageSize[1]);

    Eigen::Matrix3f K;
    K <<
        f, 0, imageSize[0] / 2.0f,
        0, f, imageSize[1] / 2.0f,
        0, 0, 1;

    Eigen::Matrix3f Rl;
    Rl =
        Eigen::AngleAxisf(0,  Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(zl, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(yl, Eigen::Vector3f::UnitY());

    Eigen::Matrix3f Rr;
    Rr =
        Eigen::AngleAxisf(xr, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(zr, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(yr, Eigen::Vector3f::UnitY());

    auto Kinverse = K.inverse();

    ml = K * Rl * Kinverse;
    mr = K * Rr * Kinverse;
}

void Rectification::paramsToMat(
        const TransformParams& params,
        int inputWidth,
        int inputHeight,
        int outputWidth,
        int outputHeight,
        Eigen::Matrix3f& ml,
        Eigen::Matrix3f& mr) const {
    paramsToMat(transform, ml, mr);

    Eigen::Matrix<float, 3, 4> cornersLeft;
    cornersLeft <<
        0.0, 0.0        , inputWidth , inputWidth,
        0.0, inputHeight, inputHeight, 0.0,
        1.0, 1.0        , 1.0        , 1.0;

    Eigen::Matrix<float, 3, 4> cornersRight;
    cornersRight <<
        0.0, 0.0        , inputWidth , inputWidth,
        0.0, inputHeight, inputHeight, 0.0,
        1.0, 1.0        , 1.0        , 1.0;

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
        Eigen::AlignedScaling2f(1.0, targetHeight / (minmaxY - maxminY)) *
        Eigen::Translation2f(0, -maxminY);

    ml = verticalTransform *
        Eigen::AlignedScaling2f(
                outputWidth / (maxLeft[0] - minLeft[0]),
                1.0) *
        Eigen::Translation2f(-minLeft[0], 0) *
        ml;

    mr = verticalTransform *
        Eigen::AlignedScaling2f(
                outputHeight / (maxRight[0] - minRight[0]),
                1.0) *
        Eigen::Translation2f(-minRight[0], 0) *
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
