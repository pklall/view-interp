#include "depth_reconstruction.h"

#include "Eigen/Dense"
#include "Eigen/SVD"

void ReconstructUtil::computeCanonicalPose(
        const Eigen::Matrix3d& E,
        array<Eigen::Matrix<double, 3, 4>, 4>& candidates) {
    // See http://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf
    // (Multiple View Geometry in Computer Vision, Second Edition,
    //  Richard Hartley and Andrew Zisserman, 2004)
    Eigen::Matrix3d W;
    W <<
        0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    Eigen::JacobiSVD<Eigen::Matrix3d> ESVD(E,
            Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix3d& U = ESVD.matrixU();
    const Eigen::Matrix3d& V = ESVD.matrixV();

    const Eigen::Vector3d& u3 = U.col(2);

    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d su3 = u3 * (((i & 0x1) == 0) ? -1 : 1);

        Eigen::Matrix3d r;
        
        if ((i & 0x2) == 0) {
            r = U * W.transpose() * V.transpose();
        } else {
            r = U * W * V.transpose();
        }

        candidates[i] <<
            r(0, 0), r(0, 1), r(0, 2), su3(0),
            r(1, 0), r(1, 1), r(1, 2), su3(1),
            r(2, 0), r(2, 1), r(2, 2), su3(2);
    }
}

void DepthReconstruction::init(
        int numCameras,
        int numPoints) {
    cameras.resize(numCameras);
    points.resize(numPoints);

    for (CameraParam& cam : cameras) {
        get<0>(cam) = Eigen::Quaterniond(1, 0, 0, 0);
        get<1>(cam) = Eigen::Vector3d(0, 0, 0);
    }

    for (Eigen::Vector3d& pt : points) {
        pt = Eigen::Vector3d(0, 0, 1.0);
    }

    ceres::Problem::Options pOptions;
    // pOptions.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    pOptions.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    problem.reset(new ceres::Problem(pOptions));

    lossFunc.reset(new ceres::LossFunctionWrapper(
                new ceres::TrivialLoss(), ceres::TAKE_OWNERSHIP));
}

void DepthReconstruction::solve(
        double huberCoeff) {

    if (huberCoeff < 0) {
        lossFunc->Reset(new ceres::TrivialLoss(), ceres::TAKE_OWNERSHIP);
    } else {
        lossFunc->Reset(new ceres::HuberLoss(huberCoeff), ceres::TAKE_OWNERSHIP);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);
    cout << summary.FullReport() << endl;
}

void DepthReconstruction::estimateNewCamera(
        int cameraIndex,
        const vector<tuple<int, Eigen::Vector2d>>& observations,
        double inlierThreshold) {
    typedef ceres::AutoDiffCostFunction<CameraReprojectionError, 2, 3, 4>
        CamCostFunction;

    // Create a new cost function for each observation
    vector<unique_ptr<CamCostFunction>> costFunctions;

    for (const tuple<int, Eigen::Vector2d>& observation : observations) {
        CamCostFunction* costFunction =
            new ceres::AutoDiffCostFunction<
            // 2 residuals
            // 3 parameters in block 1 (translation)
            // 4 parameters in block 2 (rotation)
            CameraReprojectionError, 2, 3, 4>(
                    new CameraReprojectionError(
                        this,
                        get<0>(observation),
                        (double) get<1>(observation)[0],
                        (double) get<1>(observation)[1]));

        costFunctions.push_back(unique_ptr<CamCostFunction>(costFunction));
    }

    std::random_shuffle(costFunctions.begin(), costFunctions.end());

    // Keep track of the currently-optimal score and value
    int optimalInlierCount = 0;
    CameraParam optimalParam;

    ceres::Problem::Options pOptions;
    // This ensure sthat 
    pOptions.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    // Perform 10 iterations of RANSAC
    // FIXME compute this based on acceptable probability of success
    const int max_iters = 10;

    for (int iter = 0; iter < max_iters; iter++) {
        ceres::Problem problem(pOptions);

        CameraParam curParam;

        // Use 7 correspondences to solve for camera orientation
        for (int i = 0; i < 7; i++) {
            ceres::CostFunction* cf =
                costFunctions[(iter * 7 + i) % costFunctions.size()].get();
            problem.AddResidualBlock(
                    cf,   
                    NULL, // least-squares
                    // translation
                    get<1>(curParam).data(),
                    // rotation
                    get<0>(curParam).coeffs().data());
        }

        ceres::Solver::Options options;
        // options.linear_solver_type = ???;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;

        int inlierCount = 0;

        for (const tuple<int, Eigen::Vector2d>& obs : observations) {
            double residuals[2];

            Eigen::Vector3d point3 = get3DPoint(get<0>(obs));

            computeError(
                    get<1>(curParam).data(),
                    get<0>(curParam).coeffs().data(),
                    point3.data(),
                    get<1>(obs).data(),
                    residuals);

            if (residuals[0] * residuals[0] + residuals[1] * residuals[1] < 
                    inlierThreshold * inlierThreshold) {
                inlierCount++;
            }
        }

        if (inlierCount > optimalInlierCount) {
            optimalInlierCount = inlierCount;
            optimalParam = curParam;
        }
    }

    cameras[cameraIndex] = optimalParam;
}
