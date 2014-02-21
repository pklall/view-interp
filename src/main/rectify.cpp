#include "rectify.h"

Rectification::Rectification(
        const ChainFeatureMatcher* _features,
        Eigen::Vector2i _imageSize) :
    features(_features), 
    imageSize(_imageSize) {
}

void Rectification::print(
        ostream& result) const {
    Eigen::Matrix3f ml, mr;

    paramsToMat(transform, ml, mr);

    result << "M_left = [" << endl;

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            result << ml(x, y) << "\t";
        }

        result << endl;
    }

    result << "]" << endl;

    result << "M_right = [" << endl;

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            result << mr(x, y) << "\t";
        }

        result << endl;
    }

    result << "]" << endl;
}

void Rectification::solve(
        int numRestarts) {
    residual = std::numeric_limits<double>::max();

    const int matchesPerRestart = 10;

    const vector<vector<tuple<int, Eigen::Vector2f>>>& observations =
        features->getObservations();

    vector<tuple<Eigen::Vector2f, Eigen::Vector2f>> matches(observations[0].size());

    for (const auto& obs : observations[0]) {
        matches[get<0>(obs)] = make_tuple(
                get<1>(obs),
                Eigen::Vector2f::Zero());
    }

    // TODO Handle case in which not all matches from image 0's match list
    //      have a corresponding element for image 1.
    for (const auto& obs : observations[1]) {
        int ptIndex = get<0>(obs);

        if (ptIndex < matches.size()) {
            get<1>(matches[get<0>(obs)]) = get<1>(obs);
        }
    }

    // std::random_shuffle(matches.begin(), matches.end());

    function<void(int, Eigen::Vector2f&, Eigen::Vector2f&)> pairGenAll =
        [&matches](
                int sample,
                Eigen::Vector2f& left,
                Eigen::Vector2f& right) {

            const auto& m = matches[sample];

            left = get<0>(m);
            right = get<1>(m);
        };

    TransformParams curTransform;

    unique_ptr<ceres::Problem> globalProblem =
        createProblem(pairGenAll, matches.size(), 3.0f, curTransform);

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;

    for (int i = 0; i < numRestarts; i++) {
        // Randomly shuffle if we've exhausted the already-shuffled list
        if (((i + 1) * matchesPerRestart) % matches.size() <
                matchesPerRestart) {
            std::random_shuffle(matches.begin(), matches.end());
        }

        // Clear the initial parameters
        curTransform.fill(0.0);
        
        double curResidual;

        // Fit an initial transform to a selected few points
        function<void(int, Eigen::Vector2f&, Eigen::Vector2f&)> pairGen =
            [&matches, i, matchesPerRestart](
                    int sample,
                    Eigen::Vector2f& left,
                    Eigen::Vector2f& right) {

                    const auto& m = matches[
                        (i * matchesPerRestart + sample) % matches.size()];

                    left = get<0>(m);
                    right = get<1>(m);
                };

        unique_ptr<ceres::Problem> localProblem =
            createProblem(pairGen, matchesPerRestart, -1.0f, curTransform);

        ceres::Solve(options, localProblem.get(), &summary);

        ceres::Solve(options, globalProblem.get(), &summary);

        curResidual = summary.final_cost;
        
        if (curResidual < residual) {
            residual = curResidual;

            transform = curTransform;

            printf("Found improved transform with residual: %f\n", curResidual);
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
        Eigen::AngleAxisf(yl, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(zl, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());

    Eigen::Matrix3f Rr;
    Rr =
        Eigen::AngleAxisf(yr, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(zr, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(xr, Eigen::Vector3f::UnitX());

    auto Kinverse = K.inverse();

    ml = K * Rl * Kinverse;
    mr = K * Rr * Kinverse;
}

unique_ptr<ceres::Problem> Rectification::createProblem(
        function<void(int, Eigen::Vector2f&, Eigen::Vector2f&)> pairGen,
        int numPairs,
        float robustThresh,
        TransformParams& params) const {
    unique_ptr<ceres::Problem> problem(new ceres::Problem());

    ceres::LossFunction* robustLoss = NULL;

    if (robustThresh > 0.0f) {
        robustLoss = new ceres::HuberLoss(robustThresh);
    }

    for (int i = 0; i < numPairs; i++) {
        TransformErrorFunction* func = new TransformErrorFunction();

        func->width = imageSize[0];
        func->height = imageSize[1];

        pairGen(i, func->left, func->right);

        typedef ceres::AutoDiffCostFunction<TransformErrorFunction, 1, 6>
            AutoDiffErrorFunc;

        ceres::CostFunction* costFunction = new AutoDiffErrorFunc(func);

        problem->AddResidualBlock(
                costFunction,
                robustLoss,
                params.data());
    }

    return problem;
}
