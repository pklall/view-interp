#include "reconstruct.h"

ChainReconstruction::ChainReconstruction() :
    numPoints(0) {
    prevPtGlobalPt = unique_ptr<map<int, int>>(new map<int, int>());
    curPtGlobalPt = unique_ptr<map<int, int>>(new map<int, int>());
}

void ChainReconstruction::processNext(
        const CImg<uint8_t>& gray) {
    if (!curMatcher) {
        curMatcher = unique_ptr<CVFeatureMatcher>(new CVFeatureMatcher(maxFeatureCount));
    }

    curMatcher->detectFeatures(gray);

    curPtGlobalPt->clear();

    vector<tuple<int, int>> matchBuf;

    if (prevMatcher) {
        prevMatcher->match(*curMatcher, matchBuf, maxMatchCount);

        if (matches.size() == 0) {
            matches.push_back(vector<tuple<int, float, float>>());
        }

        matches.push_back(vector<tuple<int, float, float>>());

        vector<tuple<int, float, float>>& prevMatches = matches[matches.size() - 2];
        vector<tuple<int, float, float>>& curMatches = matches[matches.size() - 1];

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

            float x, y;

            if (isNewPoint) {
                prevMatcher->getKeypoint(prevPtIndex, x, y);

                prevMatches.push_back(make_tuple(globalPtIndex, x, y));
            }

            curMatcher->getKeypoint(curPtIndex, x, y);
            curMatches.push_back(make_tuple(globalPtIndex, x, y));
        }
    }

    swap(curMatcher, prevMatcher);
    swap(curPtGlobalPt, prevPtGlobalPt);
}

void ChainReconstruction::visualizeFeatureMatches(
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

        map<int, tuple<float, float>> prevMatchMap;

        for (const auto& match : matches[i]) {
            prevMatchMap[get<0>(match)] = make_tuple(get<1>(match), get<2>(match));
        }

        int matchI = 0;
        for (const auto& match : matches[i + 1]) {
            int pt = get<0>(match);

            float curX = get<1>(match);
            float curY = get<2>(match);

            if (prevMatchMap.count(pt) > 0) {
                uint8_t color[3];

                for (int c = 0; c < 3; c++) {
                    color[c] = col(matchI % 256, c);
                }

                const auto& prevXY = prevMatchMap[pt];

                float prevX = get<0>(prevXY);
                float prevY = get<1>(prevXY);

                annotation.draw_line(
                        (int) (prevX + 0.5f),
                        (int) (prevY + 0.5f),
                        (int) (curX + 0.5f) + prevImg->width(),
                        (int) (curY + 0.5f),
                        color);

                matchI++;
            }
        }

        annotation.display();
        
        swap(prevImg, curImg);
    }
}

void ChainReconstruction::solve() {
    cameras.resize(matches.size());

    // Set initial camera values
    int camI = 0;
    for (CameraParam& cam : cameras) {
        cam[0] = 0;
        cam[1] = 0;
        cam[2] = 0;
        cam[3] = 1;
        cam[4] = (double) rand() / RAND_MAX;
        cam[5] = (double) rand() / RAND_MAX;
        cam[6] = (double) rand() / RAND_MAX;
        cam[7] = 1.0;
        cam[8] = 0.0;
        cam[9] = 0.0;

        camI++;
    }

    points.resize(numPoints);

    for (Point3d& p : points) {
        p[0] = (double) rand() / RAND_MAX;
        p[1] = (double) rand() / RAND_MAX;
        p[2] = (double) rand() / RAND_MAX;
    }

    ceres::Problem problem;

    ceres::LossFunction* lossFunc = new ceres::HuberLoss(1.0);

    for (int camI = 0; camI < matches.size(); camI++) {
        const vector<tuple<int, float, float>>& camMatches = matches[camI];

        for (const tuple<int, float, float>& match : camMatches) {
            ceres::CostFunction* costFunction = 
                new ceres::AutoDiffCostFunction<
                ceres::examples::SnavelyReprojectionErrorWithQuaternions, 2, 4, 6, 3>(
                        new ceres::examples::SnavelyReprojectionErrorWithQuaternions(
                            (double) get<1>(match),
                            (double) get<2>(match)));
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
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

