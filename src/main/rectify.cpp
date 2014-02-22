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

    // Use the top 25% as inliers
    int inlierCount = (costs.size()) * 0.25f;

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
        for (int t = 0; t < errorTerms.size(); t++) {
            get<1>(costs[t]) = t;

            double *const paramBlock[1] = {curTransform.data()};

            errorTerms[t]->Evaluate(
                    paramBlock,
                    &get<0>(costs[t]),
                    nullptr);
        }

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
