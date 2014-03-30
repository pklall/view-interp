#include "reconstruct.h"

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

ChainFeatureMatcher::ChainFeatureMatcher(
        float _maxFeatureCount,
        float _maxMatchCount) :
        maxFeatureCount(_maxFeatureCount),
        maxMatchCount(_maxMatchCount),
        numPoints(0) {
    prevPtGlobalPt = unique_ptr<map<int, int>>(new map<int, int>());
    curPtGlobalPt = unique_ptr<map<int, int>>(new map<int, int>());
}

void ChainFeatureMatcher::processNext(
        const CImg<uint8_t>& gray) {
    if (!curMatcher) {
        curMatcher = unique_ptr<CVFeatureMatcher>(new CVFeatureMatcher(maxFeatureCount, 31));
    }

    int numFeatures = curMatcher->detectFeatures(gray);

    curPtGlobalPt->clear();

    vector<tuple<int, int>> matchBuf;

    if (prevMatcher) {
        prevMatcher->match(*curMatcher, matchBuf, maxMatchCount);

        if (matches.size() == 0) {
            matches.push_back(vector<tuple<int, Eigen::Vector2f>>());
        }

        matches.push_back(vector<tuple<int, Eigen::Vector2f>>());

        vector<tuple<int, Eigen::Vector2f>>& prevMatches = matches[matches.size() - 2];
        vector<tuple<int, Eigen::Vector2f>>& curMatches = matches[matches.size() - 1];

        for (const tuple<int, int>& match : matchBuf) {
            int prevPtIndex = get<0>(match);
            int curPtIndex = get<1>(match);

            int globalPtIndex;
            
            bool isNewPoint = (prevPtGlobalPt->count(prevPtIndex) == 0);

            if (isNewPoint) {
                globalPtIndex = numPoints;

                numPoints++;
            } else {
                globalPtIndex = (*prevPtGlobalPt)[prevPtIndex];
            }

            (*prevPtGlobalPt)[prevPtIndex] = globalPtIndex;
            (*curPtGlobalPt)[curPtIndex] = globalPtIndex;

            Eigen::Vector2f point;

            if (isNewPoint) {
                prevMatcher->getKeypoint(prevPtIndex, point[0], point[1]);

                prevMatches.push_back(make_tuple(globalPtIndex, point));
            }

            curMatcher->getKeypoint(curPtIndex, point[0], point[1]);
            
            curMatches.push_back(make_tuple(globalPtIndex, point));
        }
    }

    swap(curMatcher, prevMatcher);
    swap(curPtGlobalPt, prevPtGlobalPt);
}

void ChainFeatureMatcher::visualizeFeatureMatches(
        function<const CImg<uint8_t>&(int)> imgLoader) const {
    const CImg<uint8_t>* prevImg = NULL;
    const CImg<uint8_t>* curImg = NULL;

    prevImg = &imgLoader(0);

    for (int i = 0; i < matches.size() - 1; i++) {
        curImg = &imgLoader(i + 1);
        
        assert(prevImg != NULL);
        assert(curImg != NULL);

        CImg<uint8_t> annotation(
                prevImg->width() + curImg->width(),
                max(prevImg->height(), curImg->height()),
                1, 1);

        annotation.draw_image(0, 0, 0, 0, *prevImg);

        annotation.draw_image(prevImg->width(), 0, 0, 0, *curImg);

        annotation.resize(-100, -100, -100, 3);

        CImg<uint8_t> col = CImg<uint8_t>::lines_LUT256();

        map<int, Eigen::Vector2f> prevMatchMap;

        for (const auto& match : matches[i]) {
            prevMatchMap[get<0>(match)] = get<1>(match);
        }

        int matchI = 0;
        for (const auto& match : matches[i + 1]) {
            int pt = get<0>(match);

            const auto& curXY = get<1>(match);

            if (prevMatchMap.count(pt) > 0) {
                uint8_t color[3];

                for (int c = 0; c < 3; c++) {
                    color[c] = col(matchI % 256, c);
                }

                const auto& prevXY = prevMatchMap[pt];

                annotation.draw_line(
                        (int) (prevXY[0] + 0.5f),
                        (int) (prevXY[1] + 0.5f),
                        (int) (curXY[0] + 0.5f) + prevImg->width(),
                        (int) (curXY[1] + 0.5f),
                        color);

                matchI++;
            }
        }

        annotation.display();
        
        swap(prevImg, curImg);
    }
}

ChainReconstruction::ChainReconstruction(
        const ChainFeatureMatcher* _features) :
        features(_features) {
}

void ChainReconstruction::solve() {
    const auto& matches = features->getObservations();

    int numPoints = features->getNumPoints();

    cameras.resize(matches.size());

    // Set initial camera values
    int camI = 0;
    for (CameraParam& cam : cameras) {
        cam[0] = 1;
        cam[1] = 0;
        cam[2] = 0;
        cam[3] = 0;
        cam[4] = (((double) rand()) / RAND_MAX) + (camI * 3);
        cam[5] = ((double) rand()) / RAND_MAX;
        cam[6] = ((double) rand()) / RAND_MAX;
        cam[7] = 1.0;
        cam[8] = 0.0;
        cam[9] = 0.0;

        camI++;
    }

    points.resize(numPoints);

    for (auto& p : points) {
        p[0] = (double) rand() / RAND_MAX;
        p[1] = (double) rand() / RAND_MAX;
        p[2] = (double) rand() / RAND_MAX - 10;
    }

    ceres::Problem problem;

    ceres::LossFunction* lossFunc = NULL; // new ceres::HuberLoss(25.0);

    for (int camI = 0; camI < matches.size(); camI++) {
        const vector<tuple<int, Eigen::Vector2f>>& camMatches = matches[camI];

        for (const tuple<int, Eigen::Vector2f>& match : camMatches) {
            ceres::CostFunction* costFunction = 
                new ceres::AutoDiffCostFunction<
                ceres::examples::SnavelyReprojectionErrorWithQuaternions, 2, 4, 6, 3>(
                        new ceres::examples::SnavelyReprojectionErrorWithQuaternions(
                            (double) get<1>(match)[0],
                            (double) get<1>(match)[1]));
            problem.AddResidualBlock(
                    costFunction,
                    lossFunc, // Squared loss
                    cameras[camI].data(),
                    cameras[camI].data() + 4,
                    points[get<0>(match)].data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 10000;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

void ChainReconstruction::exportPython(
    ostream& result) const {

    // Points

    result << "points = [\n";

    for (const auto& p : points) {
        result << "("
            << p[0] << ", " << p[1] << ", " << p[2]
            << "),\n";
    }

    result << "]\n";
    
    // Camera Translation

    result << "cameras_trans = [\n";

    for (const CameraParam& c : cameras) {
        result << "("
            << c[4] << ", " << c[5] << ", " << c[6]
            << "),\n";
    }

    result << "]\n";

    // Camera Rotation
    
    result << "cameras_rot = [\n";

    double mat[9];

    for (const CameraParam& c : cameras) {
        ceres::QuaternionToRotation(c.data(), mat);

        result << "(";

        for (int i = 0; i < 9; i++) {
            result << mat[i] << ", ";
        }

        result << "),\n";
    }

    result << "]\n";

    // Camera Intrinsics

    result << "cameras_intrin = [\n";

    for (const CameraParam& c : cameras) {
        result << "("
            << c[7] << ", " << c[8] << ", " << c[9]
            << "),\n";
    }

    result << "]\n";

    // Raw camera parameters

    result << "camera_raw = [\n";

    for (const CameraParam& c : cameras) {
        result << "(";

        for (auto p : c) {
            result << p << ", ";
        }

        result << "),\n";
    }

    result << "]\n";
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
