#include "depth_reconstruction.h"

#include "Eigen/Dense"
#include "Eigen/SVD"

#include "cvutil/cvutil.h"

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
    Fmatrices.resize(numCameras);
    cameras.resize(numCameras);
    observations.resize(numCameras);
    observationInlierMask.resize(numCameras);

    keypoints.resize(numPoints);
    depth.resize(numPoints);

    for (Eigen::Vector2d& pt : keypoints) {
        pt = Eigen::Vector2d(0, 0);
    }
}

void DepthReconstruction::solve() {
    resetSolutionState();

    const size_t minInliers = 200;

    float bestInlierRatio = 0;
    size_t bestCamera = 0;


    // FIXME debugging code
    CImg<float> depthVis(512, 512);
    
    for (size_t cameraI = 0; cameraI < cameras.size(); cameraI++) {
        cout << "Estimating camera " << cameraI << endl;

        size_t fInlierCount = estimateFUsingObs(cameraI);

        cout << "Inliers after computing F = " << fInlierCount << endl;

        size_t poseInlierCount = estimatePoseUsingF(cameraI);

        cout << "Inliers after computing Pose = " << poseInlierCount << endl;

        // If the pose estimated from F results in many more outliers,
        // due to negatively-facing points, it's probably a bad fit.
        float inlierRatio = poseInlierCount / fInlierCount;

        if (poseInlierCount > minInliers && inlierRatio > bestInlierRatio) {
            bestInlierRatio = inlierRatio;
            bestCamera = cameraI;
        }
    }

    /*
    size_t triCount = triangulateDepthUsingPose(bestCamera);

    cout << "Triangulation count = " << triCount << endl;

    // cout << "\n\nInitial triangulation via RANSASC-estimated F" << endl;
    // visualize(depthVis, 0, 0.75f, 2.0f, false);
    // depthVis.display();

    vector<bool> cameraMask(cameras.size());

    fill(cameraMask.begin(), cameraMask.end(), false);

    cameraMask[bestCamera] = true;

    refineCamerasAndDepth(cameraMask);
    */

    // cout << "\n\nAfter refinement" << endl;
    // visualize(depthVis, 0, 0.75f, 2.0f, false);
    // depthVis.display();

    // FIXME don't use this testing code
    /*
    for (size_t cameraI = 0; cameraI < cameras.size(); cameraI++) {
        std::fill(
                observationInlierMask[cameraI].begin(),
                observationInlierMask[cameraI].end(),
                true);
        size_t poseInlierCount = estimatePoseUsingDepth(cameraI, 0.0005);
    }

    for (size_t cameraI = 0; cameraI < cameras.size(); cameraI++) {
        printf("Camera %d\n", cameraI);

        std::fill(depth.begin(), depth.end(), 0);

        size_t triCount = triangulateDepthUsingPose(cameraI);

        cout << "\n\nTriangulation after depth-based pose estimation" << endl;
        visualize(depthVis, 0, 0.75f, 2.0f, false);
        depthVis.display();

        cameraMask[cameraI] = true;

        refineCamerasAndDepth(cameraMask);

        cout << "\n\nAfter refinement" << endl;
        visualize(depthVis, 0, 0.75f, 2.0f, false);
        depthVis.display();
    }
    */

    /*
    for (size_t cameraI = 0; cameraI < cameras.size(); cameraI++) {
        if (cameraI == bestCamera) {
            // Don't re-process the initial camera
            continue;
        }

        resetInlierMask(cameraI);

        size_t poseInlierCount = estimatePoseUsingDepth(cameraI, 0.01);

        cout << "Inliers after computing Pose = " << poseInlierCount << endl;

        if (poseInlierCount > 200) {
            size_t triCount = triangulateDepthUsingPose(cameraI);

            cout << "Triangulation count = " << triCount << endl;

            cameraMask[cameraI] = true;

            refineCamerasAndDepth(cameraMask);

            cout << "\n\nAfter refinement" << endl;
            visualize(depthVis, 0, 0.75f, 2.0f, false);
            depthVis.display();
        }
    }
    */

    // Set inlierCount to tally the total number of inlier observations
    // used in determining each depth value.
    for (size_t cameraI = 0; cameraI < cameras.size(); cameraI++) {
        const vector<Observation> obs = observations[cameraI];
        const vector<bool> inlierMask = observationInlierMask[cameraI];

        for (size_t obsI = 0; obsI < obs.size(); obsI++) {
            if (inlierMask[obsI]) {
                inlierCount[obs[obsI].pointIndex]++;
            }
        }
    }
}

void DepthReconstruction::visualize(
        CImg<float>& depthVis,
        int minInlierCount,
        float inlierPercentile,
        float inlierRangeMultiplier,
        bool printToStdout = false) {
    depthVis.fill(0);
    float totalDepth = 0;

    int successfulMatches = 0;

    const double imageSize = max(depthVis.width(), depthVis.height()) / 2.0;

    Eigen::Vector2d imageCenter(
            depthVis.width() / 2.0,
            depthVis.height() / 2.0);

    vector<double> depthSamples;

    for (size_t i = 0; i < getPointCount(); i++) {
        double depth;
        size_t inlierCount;

        getDepthSample(i, depth, inlierCount);

        if (depth > 0 && inlierCount >= minInlierCount) {
            depthSamples.push_back(depth);
        }
    }

    size_t inlierDepthIndex = depthSamples.size() * inlierPercentile;

    inlierDepthIndex = max((size_t) 0, inlierDepthIndex);
    inlierDepthIndex = min(depthSamples.size() - 1, inlierDepthIndex);

    if (depthSamples.size() > 0) {
        std::nth_element(
                &(depthSamples[0]),
                &(depthSamples[inlierDepthIndex]),
                &(depthSamples[depthSamples.size() - 1]));
    }

    double maxDepth = depthSamples[inlierDepthIndex] * inlierRangeMultiplier;

    for (size_t i = 0; i < getPointCount(); i++) {
        double depth;
        size_t inlierCount;

        const Eigen::Vector2d& pt = getDepthSample(i, depth, inlierCount);

        if (depth > 0 &&
                inlierCount >= minInlierCount &&
                depth < maxDepth) {
            Eigen::Vector2d ptImg = (pt * imageSize) + imageCenter;

            totalDepth += depth;
            successfulMatches++;

            depthVis.draw_circle(ptImg.x() + 0.5, ptImg.y() + 0.5, 3, &depth);
        }
    }

    // Fill the background with the average depth
    float avgDepth = totalDepth / successfulMatches;
    depthVis -= (depthVis.get_sign().abs() - 1) * avgDepth;

    if (printToStdout) {
        printf("\n\n[");

        for (size_t i = 0; i < getPointCount(); i++) {
            double depth;
            size_t inlierCount;

            const Eigen::Vector2d& pt = getDepthSample(i, depth, inlierCount);

            if (depth > 0 &&
                    inlierCount >= minInlierCount &&
                    depth < maxDepth) {
                printf("(%f, %f, %f),", pt.x(), pt.y(), depth);
            }
        }

        printf("]\n\n");
    }
}

bool DepthReconstruction::getPolarFundamentalMatrix(
        int cameraIndex,
        const Eigen::Vector2d& imageCenter,
        double imageSize,
        PolarFundamentalMatrix& polarF) {
    Eigen::Matrix3d E = getEssentialMatrix(cameraIndex);

    Eigen::Matrix3d K = (
            Eigen::Translation2d(imageCenter) *
            Eigen::Scaling(imageSize)).matrix();

    Eigen::Matrix3d Kinv = (
            Eigen::Scaling(1.0 / imageSize) *
            Eigen::Translation2d(-imageCenter)).matrix();

    Eigen::Matrix3d F = Kinv.transpose() * E * Kinv;

    array<Eigen::Vector2d, 2> match;

    if (observations[cameraIndex].size() == 0) {
        return false;
    }

    const Observation& obs = observations[cameraIndex][0];

    Eigen::Vector3d match0H = Kinv * keypoints[obs.pointIndex].homogeneous();
    Eigen::Vector3d match1H = Kinv * obs.point.homogeneous();

    match[0] = match0H.hnormalized();
    match[1] = match1H.hnormalized();

    return polarF.init(F, match);
}

void DepthReconstruction::resetSolutionState() {
    std::fill(depth.begin(), depth.end(), 0);

    inlierCount.resize(depth.size());
    std::fill(inlierCount.begin(), inlierCount.end(), 0);

    for (size_t cameraI = 0; cameraI < cameras.size(); cameraI++) {
        vector<bool>& inlierMask = observationInlierMask[cameraI];

        inlierMask.resize(observations[cameraI].size());

        // Mark all observations as inliers
        std::fill(inlierMask.begin(), inlierMask.end(), true);
    }
}

void DepthReconstruction::resetInlierMask(
        int cameraIndex) {
    vector<bool>& inlierMask = observationInlierMask[cameraIndex];

    std::fill(inlierMask.begin(), inlierMask.end(), true);
}

size_t DepthReconstruction::estimateFUsingObs(
        int cameraIndex) {
    fundMatEstimator.init();

    for (const Observation& obs : observations[cameraIndex]) {
        fundMatEstimator.addMatch(keypoints[obs.pointIndex], obs.point);
    }

    fundMatEstimator.estimateFundamentalMatrix(Fmatrices[cameraIndex]);

    size_t inlierCount = 0;

    for (size_t i = 0; i < observations[cameraIndex].size(); i++) {
        if (fundMatEstimator.isInlier(i)) {
            inlierCount++;

            observationInlierMask[cameraIndex][i] = true;
        } else {
            observationInlierMask[cameraIndex][i] = false;
        }
    }

    return inlierCount;
}

size_t DepthReconstruction::estimatePoseUsingF(
        int cameraIndex) {
    // Candidate projection matrices for the current camera
    array<Eigen::Matrix<double, 3, 4>, 4> candidateP;

    ReconstructUtil::computeCanonicalPose(Fmatrices[cameraIndex], candidateP);

    // Each observation votes for the correct pose of the 4-possible candidates
    // The 5th ballot is reserved for stupid voters who should be eliminated
    // for not fitting in with the system.
    array<size_t, 5> poseBallots;
    poseBallots.fill(0);

    for (size_t obsI = 0; obsI < observations[cameraIndex].size(); obsI++) {
        const Observation& obs = observations[cameraIndex][obsI];

        // Only inliers get to vote
        if (observationInlierMask[cameraIndex][obsI]) {
            int vote = ReconstructUtil::selectCandidatePose(
                    keypoints[obs.pointIndex], obs.point, candidateP);

            if (vote > 0) {
                poseBallots[vote]++;
            } else {
                poseBallots[4]++;
            }
        }
    }

    int winner = std::max_element(poseBallots.begin(), poseBallots.end() - 1) -
        poseBallots.begin();

    const Eigen::Matrix<double, 3, 4>& pose = candidateP[winner];

    const auto rotation = pose.block<3, 3>(0, 0);

    const auto translation = pose.block<3, 1>(0, 3);

    cameras[cameraIndex].rotation = Eigen::Quaterniond(rotation).normalized();
    cameras[cameraIndex].translation = translation;

    // Count inliers and mark new outliers

    size_t inlierCount = 0;

    for (size_t obsI = 0; obsI < observations[cameraIndex].size(); obsI++) {
        const Observation& obs = observations[cameraIndex][obsI];

        if (observationInlierMask[cameraIndex][obsI]) {
            int vote = ReconstructUtil::selectCandidatePose(
                    keypoints[obs.pointIndex], obs.point, candidateP);

            if (vote == winner) {
                inlierCount++;
            } else {
                observationInlierMask[cameraIndex][obsI] = false;
            }
        }
    }

    return inlierCount;
}

size_t DepthReconstruction::triangulateDepthUsingPose(
        int cameraIndex) {
    const auto& P = cameras[cameraIndex].getP();

    size_t newSampleCount = 0;
    size_t reverseCount = 0;

    for (size_t obsI = 0; obsI < observations[cameraIndex].size(); obsI++) {
        const Observation& obs = observations[cameraIndex][obsI];

        if (observationInlierMask[cameraIndex][obsI] &&
                depth[obs.pointIndex] == 0) {
            double d = ReconstructUtil::triangulateDepth(
                    keypoints[obs.pointIndex], obs.point, P);

            if (d > 0 && isfinite(d)) {
                depth[obs.pointIndex] = d;

                newSampleCount++;
            } else {
                reverseCount++;
            }
        }
    }

    cout << "Reverse = " << reverseCount << endl;

    return newSampleCount;
}

size_t DepthReconstruction::estimatePoseUsingDepth(
        int cameraIndex,
        double inlierThreshold) {
    typedef ceres::AutoDiffCostFunction<CameraReprojectionError, 2, 3, 4>
        CamCostFunction;

    vector<unique_ptr<CamCostFunction>> costFunctions;

    for (size_t obsI = 0; obsI < observations[cameraIndex].size(); obsI++) {
        const Observation& obs = observations[cameraIndex][obsI];

        // Create a new cost function for each inlier observation for which
        // depth has already been estimated.
        if (observationInlierMask[cameraIndex][obsI] &&
                depth[obs.pointIndex] > 0) {

            CamCostFunction* costFunction =
                new ceres::AutoDiffCostFunction<
                // 2 residuals
                // 3 parameters in block 1 (translation)
                // 4 parameters in block 2 (rotation)
                CameraReprojectionError, 2, 3, 4>(
                        new CameraReprojectionError(this, &obs));

            costFunctions.push_back(unique_ptr<CamCostFunction>(costFunction));
        }
    }

    if (costFunctions.size() == 0) {
        return 0;
    }

    // Perform RANSAC by randomly selecting cost function terms, solving for
    // model parameters, and counting inliers from the resulting fit.
    std::random_shuffle(costFunctions.begin(), costFunctions.end());

    // Keep track of the currently-optimal score and value
    int optimalInlierCount = 0;
    CameraParam optimalParam;

    ceres::Problem::Options pOptions;
    // This ensures that ceres don't delete our cost functions
    pOptions.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    pOptions.local_parameterization_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    // Perform 10 iterations of RANSAC
    // TODO compute this based on acceptable probability of success
    const int max_iters = 1000;

    unique_ptr<ceres::LocalParameterization> quatParameterization(
            new ceres::QuaternionParameterization());

    for (int iter = 0; iter < max_iters; iter++) {
        ceres::Problem problem(pOptions);

        CameraParam curParam;

        curParam.translation = Eigen::Vector3d(0, 0, 0);
        curParam.rotation = Eigen::Quaterniond(1, 0, 0, 0);

        // Use 5 correspondences to solve for camera orientation
        // TODO What's the correct amount to use here?
        for (int i = 0; i < 5; i++) {
            ceres::CostFunction* cf =
                costFunctions[(iter * 5 + i) % costFunctions.size()].get();

            problem.AddResidualBlock(
                    cf,
                    NULL, // least-squares
                    curParam.translation.data(),
                    curParam.rotation.coeffs().data());

            problem.SetParameterization(curParam.rotation.coeffs().data(),
                    quatParameterization.get());
        }

        ceres::Solver::Options options;
        // Use default solver
        // options.linear_solver_type = ???;
        options.max_num_iterations = 100;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << endl;

        int inlierCount = 0;

        // Compute the number of inliers among those samples we care about
        for (size_t obsI = 0; obsI < observations[cameraIndex].size(); obsI++) {
            const Observation& obs = observations[cameraIndex][obsI];

            if (observationInlierMask[cameraIndex][obsI] &&
                    depth[obs.pointIndex] > 0) {
                double residuals[2];

                Eigen::Vector3d point3 = get3DPoint(obs.pointIndex);

                computeError(
                        curParam.translation.data(),
                        curParam.rotation.coeffs().data(),
                        point3.data(),
                        keypoints[obs.pointIndex].data(),
                        residuals);

                if (residuals[0] * residuals[0] + residuals[1] * residuals[1] < 
                        inlierThreshold * inlierThreshold) {
                    inlierCount++;
                }
            }
        }

        if (inlierCount > optimalInlierCount) {
            optimalInlierCount = inlierCount;
            optimalParam = curParam;
        }
    }

    // Refine with all points and a robust kernel
    {
        ceres::Problem problem(pOptions);

        // FIXME don't hardcode parameter
        ceres::LossFunction* robustLoss = new ceres::HuberLoss(0.00001);

        for (const auto& cf : costFunctions) {
            problem.AddResidualBlock(
                    cf.get(),
                    robustLoss,
                    optimalParam.translation.data(),
                    optimalParam.rotation.coeffs().data());

            problem.SetParameterization(
                    optimalParam.rotation.coeffs().data(),
                    quatParameterization.get());
        }

        ceres::Solver::Options options;
        // Use default solver
        // options.linear_solver_type = ???;
        options.max_num_iterations = 100;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    // Update inlier masks
    for (size_t obsI = 0; obsI < observations[cameraIndex].size(); obsI++) {
        const Observation& obs = observations[cameraIndex][obsI];

        if (observationInlierMask[cameraIndex][obsI] &&
                depth[obs.pointIndex] > 0) {
            double residuals[2];

            Eigen::Vector3d point3 = get3DPoint(obs.pointIndex);

            computeError(
                    optimalParam.translation.data(),
                    optimalParam.rotation.coeffs().data(),
                    point3.data(),
                    keypoints[obs.pointIndex].data(),
                    residuals);

            if (residuals[0] * residuals[0] + residuals[1] * residuals[1] < 
                    inlierThreshold * inlierThreshold) {
            } else {
                observationInlierMask[cameraIndex][obsI] = false;
            }
        }
    }

    // Since we may have solved for an incorrect (i.e. backwards-facing)
    // pose, we must have another "election" to find the correct one.
    Fmatrices[cameraIndex] = optimalParam.getE();

    return estimatePoseUsingF(cameraIndex);
}

void DepthReconstruction::refineCamerasAndDepth(
        const vector<bool>& cameraMask) {
    typedef ceres::AutoDiffCostFunction<CamDepthReprojectionError, 2, 3, 4, 1>
        CamCostFunction;

    ceres::Problem problem;
    // FIXME don't hard-code this parameter
    ceres::LossFunction* huberLoss = new ceres::HuberLoss(0.00001);
    ceres::LocalParameterization* quatParameterization(
            new ceres::QuaternionParameterization());

    for (size_t camI = 0; camI < cameras.size(); camI++) {
        if (cameraMask[camI]) {
            for (size_t obsI = 0; obsI < observations[camI].size(); obsI++) {
                const Observation& obs = observations[camI][obsI];

                // Create a new cost function for each inlier observation for which
                // depth has already been estimated.
                if (observationInlierMask[camI][obsI] &&
                        depth[obs.pointIndex] > 0) {

                    CamCostFunction* costFunction =
                        new ceres::AutoDiffCostFunction<
                        // 2 residuals
                        // 3 parameters in block 1 (translation)
                        // 4 parameters in block 2 (rotation)
                        // 1 parameter in block 3 (depth)
                        CamDepthReprojectionError, 2, 3, 4, 1>(
                                new CamDepthReprojectionError(this, &obs));

                    problem.AddResidualBlock(
                            costFunction,
                            huberLoss,
                            cameras[camI].translation.data(),
                            cameras[camI].rotation.coeffs().data(),
                            &(depth[obs.pointIndex]));

                    problem.SetParameterization(
                            cameras[camI].rotation.coeffs().data(),
                            quatParameterization);
                }
            }
        }
    }

    ceres::Solver::Options options;
    // Use default solver
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

