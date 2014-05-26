#include "tri_qpbo.h"

#include <random>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/mqpbo.hxx"
#include "opengm/inference/graphcut.hxx"
#include "opengm/inference/auxiliary/minstcutboost.hxx"

#include "boost/polygon/voronoi.hpp"

using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;

// Create required concepts and traits to use Eigen::Vector2f with boost's
// polygon library.
namespace boost {
    namespace polygon {
        template<>
            struct geometry_concept<Eigen::Vector2f> {
                typedef point_concept type;
            };

        template<>
            struct point_traits<Eigen::Vector2f> {
                typedef int coordinate_type;

                static inline coordinate_type get(
                        const Eigen::Vector2f& point,
                        boost::polygon::orientation_2d orient) {
                    if (orient == boost::polygon::HORIZONTAL) {
                        return (int) point.x();
                    } else {
                        return (int) point.y();
                    }
                }
            };
    }
}

struct TriQPBO::GModelData {
    typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
    typedef opengm::ExplicitFunction<float, size_t, size_t> ExplicitFunction;

    typedef opengm::GraphicalModel<float, opengm::Adder, ExplicitFunction,
            Space> GModel;

    typedef opengm::external::QPBO<GModel> QPBO;

    inline void setUnaryTerm(
            size_t triI,
            float cost0,
            float cost1) {
        ExplicitFunction& func =
            model.getFunction<ExplicitFunction>(unaryFIds[triI]);

        func(0) = cost0;
        func(1) = cost1;
    }

    inline void setBinaryTerm(
            size_t edgeI,
            float cost00,
            float cost01,
            float cost10,
            float cost11) {
        ExplicitFunction& func =
            model.getFunction<ExplicitFunction>(binaryFIds[edgeI]);

        func(0, 0) = cost00;
        func(0, 1) = cost01;
        func(1, 0) = cost10;
        func(1, 1) = cost11;
    }

    // Map triangle edge indices to the associated binary function identifier
    vector<GModel::FunctionIdentifier> binaryFIds;

    // Map triangle indices to the associated unary function identifier
    vector<GModel::FunctionIdentifier> unaryFIds;

    GModel model;

    unique_ptr<QPBO> qpbo;
};

TriQPBO::TriQPBO(
        const CImg<float>& lab,
        const vector<Eigen::Vector2f>& _points) :
    imgLab(lab),
    gModelData(nullptr),
    points(_points),
    adjTriCount(0) {

    vertexCandidates.resize(points.size());
}

void TriQPBO::init() {
    initTriangles();

    initTriangleLabels();

    initTriangleColorStats();

    initAdjacency();

    initTrianglePairCost();

    initGModel();
}

TriQPBO::~TriQPBO() {
    delete gModelData;
}

void TriQPBO::denseInterp(
        CImg<double>& result) {
    for (int i = 0; i < triangles.size(); i++) {
        const auto& tri = triangles[i];

        // The following based on the draw_triangle method from CImg.h
        // Note that the gouroud shading and color interpolation implementations
        // in CImg only work on 8-bit integers.
        const double x0 = points[get<0>(tri)].x();
        const double y0 = points[get<0>(tri)].y();
        const double x1 = points[get<1>(tri)].x();
        const double y1 = points[get<1>(tri)].y();
        const double x2 = points[get<2>(tri)].x();
        const double y2 = points[get<2>(tri)].y();
        const double& v = triangleValues[i];

        int nx0 = x0, ny0 = y0, nx1 = x1, ny1 = y1, nx2 = x2, ny2 = y2;
        if (ny0>ny1) cimg::swap(nx0,nx1,ny0,ny1);
        if (ny0>ny2) cimg::swap(nx0,nx2,ny0,ny2);
        if (ny1>ny2) cimg::swap(nx1,nx2,ny1,ny2);
        if (ny0<result.height() && ny2>=0) {
            _cimg_for_triangle1(result,xl,xr,y,nx0,ny0,nx1,ny1,nx2,ny2) {
                int xleft = xl;
                int xright = xr;

                if (xright < xleft) {
                    swap(xleft, xright);
                }

                for (int x = max(xleft, 0);
                        x <= min(xright, result.width() - 1);
                        x++) {
                    result(x, y) = v;
                }
            }
        }
    }
}

void TriQPBO::visualizeTriangulation(
        CImg<float>& colorVis) {
    float color[3];

    /*
    for (int triI = 0; triI < triangles.size(); triI++) {
        const auto& tri = triangles[triI];

        for (int c = 0; c < 3; c++) {
            color[c] = get<0>(triangleLabMeanVar[triI])[c];
        }

        colorVis.draw_triangle(
                points[get<0>(tri)].x(),
                points[get<0>(tri)].y(),
                points[get<1>(tri)].x(),
                points[get<1>(tri)].y(),
                points[get<2>(tri)].x(),
                points[get<2>(tri)].y(),
                color);
    }
    */

    // Draw adjacency graph as lines between the centers of triangles
    color[0] = 0;
    color[1] = 0;
    color[2] = 0;

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const auto& adj = adjacency[triI];

        const auto& tri = triangles[triI];

        double cx = 
            points[get<0>(tri)].x() +
            points[get<1>(tri)].x() +
            points[get<2>(tri)].x();
        cx /= 3.0;
        double cy =
            points[get<0>(tri)].y() +
            points[get<1>(tri)].y() +
            points[get<2>(tri)].y();
        cy /= 3.0;

        for (const auto& adjPair : adj) {
            size_t adjTriI = adjPair.first;

            size_t pairId = adjPair.second.id;

            float dist =  trianglePairCost[pairId];/* *
                min(
                        fabs(
                            triangleValues[triI] -
                            triangleValues[adjTriI]) / (2 * adjTriValueVariance),
                       1.0);
                */
            color[0] = dist;
            color[1] = dist;
            color[2] = dist;
            
            const auto& aTri = triangles[adjTriI];

            double cx2 = 
                points[get<0>(aTri)].x() +
                points[get<1>(aTri)].x() +
                points[get<2>(aTri)].x();
            cx2 /= 3.0;
            double cy2 =
                points[get<0>(aTri)].y() +
                points[get<1>(aTri)].y() +
                points[get<2>(aTri)].y();
            cy2 /= 3.0;

            colorVis.draw_line(cx, cy, cx2, cy2, color);
        }
    }
}

void TriQPBO::addCandidateVertexDepths(
        vector<double>& depths,
        bool fitLinear) {
    assert(depths.size() == points.size());

    if (fitLinear) {
        fitCandidateValuesLinear(depths);
    }

    addCandidateVertexDepths(depths);
}

void TriQPBO::addCandidateVertexDepths(
        const vector<double>& depths) {
    assert(depths.size() == points.size());

    for (size_t i = 0; i < depths.size(); i++) {
        if (depths[i] > 0 && isfinite(depths[i])) {
            vertexCandidates[i].push_back(depths[i]);
        } else {
            vertexCandidates[i].push_back(-1);
        }
    }
}

void TriQPBO::solveAlphaExpansion(
        double minDepth,
        double maxDepth,
        int numLabels,
        int numIters,
        float unaryCostFactor) {
    // vector<double> sortedTriangleValues;

    // for (const double& d : triangleValues) {
        // if (d > 0) {
            // sortedTriangleValues.push_back(d);
        // }
    // }

    // sort(sortedTriangleValues.begin(), sortedTriangleValues.end());

    // const size_t numLabels = min((size_t) 64, sortedTriangleValues.size());

    vector<double> candidates(numLabels);

    vector<vector<float>> candidateUnaryCosts(numLabels);

    for (int i = 0; i < numLabels; i++) {
        // size_t offset = sortedTriangleValues.size() * 0.25;
        // size_t num = sortedTriangleValues.size() * 0.50;
        // size_t index = i * (num / (float) numLabels) + offset;

        // candidates[i] = sortedTriangleValues[index];
        candidates[i] = minDepth +
            (maxDepth - minDepth) * ((double) i) / (double) numLabels;

        candidateUnaryCosts[i].resize(triangles.size());
    }

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        float totalCost = 0;

        for (size_t candidateI = 0; candidateI < candidates.size(); candidateI++) {

            const double& candidateLabel = candidates[candidateI];

            const array<size_t, 3>& tri = triangles[triI];

            float cost = unaryCostTri(triI, candidateLabel);

            candidateUnaryCosts[candidateI][triI] = cost;

            totalCost += cost;
        }

        // Normalize unary costs across all candidates for a particular triangle
        for (size_t candidateI = 0; candidateI < candidates.size(); candidateI++) {
            candidateUnaryCosts[candidateI][triI] /= totalCost;
        }
    }

    fill(triangleValues.begin(), triangleValues.end(), -1);

    for (int i = 0; i < numIters; i++) {
        for (size_t cI = 0; cI < candidates.size(); cI++) {
            const double& label = candidates[cI];
            
            // TODO this is a waste, mergeCandidateValues should take a single
            //      global candidate value instead
            vector<double> newTriangleValues(triangles.size());

            std::fill(newTriangleValues.begin(), newTriangleValues.end(), label);

            size_t numChanged = mergeCandidateValues(newTriangleValues,
                    candidateUnaryCosts[cI], unaryCostFactor);

            printf("Label: %f\n", label);
            printf("Number of label flips = %d\n", numChanged);
        }
    }
}

void TriQPBO::solve(
        int numIters,
        float unaryCostFactor) {
    size_t maxNumCandidates = 0;

    for (const auto& c : vertexCandidates) {
        maxNumCandidates = max(maxNumCandidates, c.size());
    }

    const int numLabels = maxNumCandidates * 3;

    vector<vector<double>> candidates(numLabels);

    vector<vector<float>> candidateUnaryCosts(numLabels);

    for (int i = 0; i < numLabels; i++) {
        candidates[i].resize(triangles.size());

        fill(candidates[i].begin(), candidates[i].end(), -1);

        candidateUnaryCosts[i].resize(triangles.size());

        fill(candidateUnaryCosts[i].begin(), candidateUnaryCosts[i].end(),
                std::numeric_limits<float>::max());
    }

    // Add a candidate triangle label for each vertex candidate
    for (size_t triI = 0; triI < triangles.size(); triI++) {
        for (int i = 0; i < 3; i++) {
            const vector<double>& vc = vertexCandidates[triangles[triI][i]];

            for (size_t j = 0; j < vc.size(); j++) {
                candidates[i + j * 3][triI] = vc[j];
            }
        }
    }

    // Compute unary costs
    for (size_t labelI = 0; labelI < numLabels; labelI++) {
        for (size_t triI = 0; triI < triangles.size(); triI++) {
            candidateUnaryCosts[labelI][triI] =
                unaryCostTri(triI, candidates[labelI][triI]);
        }
    }
    
    // Normalize unary costs with respect to all labels for a particular triangle
    /*
    for (size_t triI = 0; triI < triangles.size(); triI++) {
        float totalCost = 0;

        for (size_t candidateI = 0; candidateI < numLabels; candidateI++) {
            totalCost += candidateUnaryCosts[candidateI][triI];
        }

        for (size_t candidateI = 0; candidateI < candidates.size(); candidateI++) {
            candidateUnaryCosts[candidateI][triI] /= totalCost;
        }
    }
    */

    fill(triangleValues.begin(), triangleValues.end(), -1);

    for (int i = 0; i < numIters; i++) {
        for (size_t cI = 0; cI < candidates.size(); cI++) {
            const vector<double>& newTriangleValues = candidates[cI];

            size_t numChanged = mergeCandidateValues(newTriangleValues,
                    candidateUnaryCosts[cI],
                    unaryCostFactor);

            printf("Number of label flips = %d\n", numChanged);
        }
    }

    vector<double> newTriangleValues(triangles.size());
    vector<float> newTriangleValuesCosts(triangles.size());
    fill(newTriangleValuesCosts.begin(), newTriangleValuesCosts.end(), 0);

    for (int i = 0; i < 1; i++) {
        printf("Using adjacent labels for label set\n");

        vector<size_t> toVisit;
        set<size_t> visited;
        size_t toVisitFront = 0;
        vector<double> closeValues;

        for (size_t triI = 0; triI < triangles.size(); triI++) {
            toVisit.clear();
            toVisitFront = 0;
            visited.clear();
            closeValues.clear();

            toVisit.push_back(triI);

            for (int n = 0; n < 15; n++) {
                size_t neighbor = toVisit[toVisitFront];
                toVisitFront++;
                visited.insert(neighbor);

                const auto& adj = adjacency[neighbor];

                vector<tuple<float, size_t>> pToVisit;

                for (const auto& a : adj) {
                    if (visited.count(a.first) == 0) {
                        pToVisit.push_back(make_tuple(trianglePairCost[a.second.id],
                                    a.first));
                    }
                }

                if (pToVisit.size() > 0) {
                    sort(pToVisit.begin(), pToVisit.end());
                    toVisit.push_back(get<1>(pToVisit[0]));
                }

                closeValues.push_back(triangleValues[neighbor]);
            }

            size_t medianI = closeValues.size() / 2;
            nth_element(
                    &(closeValues.front()),
                    &(closeValues[medianI]),
                    &(closeValues.back()));
            newTriangleValues[triI] = closeValues[medianI];
            triangleValues[triI] = newTriangleValues[triI];
        }

        /*
        for (size_t triI = 0; triI < triangles.size(); triI++) {
            const auto& adj = adjacency[triI];
            const auto& tri = triangles[triI];

            float lowestCost = std::numeric_limits<float>::max();

            for (const auto& adjPair : adj) {
                size_t adjTriI = adjPair.first;

                size_t pairId = adjPair.second.id;

                const float& pairCost = trianglePairCost[pairId];

                if (pairCost < lowestCost) {
                    lowestCost = pairCost;
                    newTriangleValues[triI] = triangleValues[adjTriI];
                }
            }

            // newTriangleValues[triI] = triangleValues[tri[i % 3]];
        }
        */

        size_t numChanged = mergeCandidateValues(
                newTriangleValues,
                newTriangleValuesCosts,
                unaryCostFactor);

        printf("Number of label flips = %d\n", numChanged);
    }
}

// Functors needed for solveSmooth()...
struct ScaledBinaryEqualityResidual {
    template<typename T>
        bool operator()(
                const T* const a,
                const T* const b,
                T* residuals) const {
            residuals[0] = (a[0] - b[0]) * T(scale);
            
            return true;
        }

    float scale;
};

struct UnaryStaticResidual {
    UnaryStaticResidual(double _e) : expected(_e) {}

    template<typename T>
        bool operator()(
                const T* const val,
                T* residuals) const {
            residuals[0] = val[0] - T(expected);
            
            return true;
        }

    double expected;
};

struct WeightedLaplaceResidual {
    template<typename T>
    bool operator()(
            T const* const* values,
            T* residuals) const {
        T r = values[0];

        for (int i = 0; i < weights.size(); i++) {
            r -= values[i] * T(weights[i]);
        }

        residuals[0] = r;
    }

    vector<float> weights;
};

void TriQPBO::solveSmooth() {
    smoothTriangleValues.resize(triangles.size());

    double scale;
    vector<double> sortedTriangleValues = triangleValues;
    nth_element(
            &(sortedTriangleValues.front()),
            &(sortedTriangleValues[sortedTriangleValues.size() / 2]),
            &(sortedTriangleValues.back()));
    scale = 1.0 / sortedTriangleValues[sortedTriangleValues.size() / 2];

    ceres::Problem problem;

    map<size_t, vector<tuple<size_t, double*>>> vertexIndexToSmoothVertexValues;

    ceres::LossFunction* robustLoss =
        new ceres::HuberLoss(0.00000001);

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const array<size_t, 3> tri = triangles[triI];

        double scaledTriVal = triangleValues[triI] * scale;

        // 1 residual
        // 1 parameter block
        ceres::CostFunction* cf =
            new ceres::AutoDiffCostFunction<UnaryStaticResidual, 1, 1>(
                    new UnaryStaticResidual(scaledTriVal));

        for (int i = 0; i < 3; i++) {
            // Set initial value
            smoothTriangleValues[triI][i] = scaledTriVal;

            vertexIndexToSmoothVertexValues[tri[i]].push_back(
                    make_tuple(triI, &(smoothTriangleValues[triI][i])));

            problem.AddResidualBlock(
                    cf,
                    robustLoss,
                    &(smoothTriangleValues[triI][i]));
        }
    }

    ceres::Solver::Options options;
    // Use default solver
    // FIXME lbfgs might converge faster than the default (levenberg marquardt)
    // options.linear_solver_type = ???;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = true;


    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport();
}

void TriQPBO::solveSmoothHack() {
    smoothTriangleValues.resize(triangles.size());

    double scale;
    vector<double> sortedTriangleValues = triangleValues;
    nth_element(
            &(sortedTriangleValues.front()),
            &(sortedTriangleValues[sortedTriangleValues.size() / 2]),
            &(sortedTriangleValues.back()));
    scale = 1.0 / sortedTriangleValues[sortedTriangleValues.size() / 2];

    for (auto& v : triangleValues) {
        v *= scale;
    }

    map<size_t, vector<tuple<size_t, double*>>> vertexIndexToSmoothVertexValues;

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const array<size_t, 3> tri = triangles[triI];

        double scaledTriVal = triangleValues[triI];

        for (int i = 0; i < 3; i++) {
            // Set initial value
            smoothTriangleValues[triI][i] = scaledTriVal;

            vertexIndexToSmoothVertexValues[tri[i]].push_back(
                    make_tuple(triI, &(smoothTriangleValues[triI][i])));
        }
    }
    
    for (auto& pair : vertexIndexToSmoothVertexValues) {
        double total = 0;

        for (const auto& t : pair.second) {
            total += *get<1>(t);
        }

        total /= pair.second.size();

        for (auto& t : pair.second) {
            *get<1>(t) = total;
        }
    }
}

void TriQPBO::smoothAvg() {
    vector<double> newTriangleValues(triangleValues.size());

    double alpha = 0.5;

    for (int iter = 0; iter < 5; iter++) {
        for (size_t triI = 0; triI < triangles.size(); triI++) {
            double totalWeight = 0; 
            double totalValue = 0;

            for (const auto& pair : adjacency[triI]) {
                size_t adjTriI = pair.first;
                size_t pairIdx = pair.second.id;

                const double& pairCost = trianglePairCost[pairIdx];

                if (isfinite(triangleValues[adjTriI]) &&
                        triangleValues[adjTriI] > 0) {
                    totalWeight += pairCost;
                    totalValue += triangleValues[adjTriI] * pairCost;
                }
            }

            if (isfinite(triangleValues[triI]) && triangleValues[triI] > 0) {
                newTriangleValues[triI] = triangleValues[triI];
            }

            if (totalWeight > 0) {
                newTriangleValues[triI] = newTriangleValues[triI] * alpha +
                    (1.0 - alpha) * totalValue / totalWeight;
            }
        }

        swap(triangleValues, newTriangleValues);
    }
}

size_t TriQPBO::mergeCandidateValues(
        const vector<double>& candidates,
        const vector<float>& unaryCosts,
        float unaryCostFactor) {
    // Update the factor graph...
    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const auto& adj = adjacency[triI];
        const auto& tri = triangles[triI];


        float totalPairCostFactor = 0;

        // Update binary terms
        for (const auto& adjPair : adj) {
            size_t adjTriI = adjPair.first;

            size_t pairId = adjPair.second.id;

            totalPairCostFactor += trianglePairCost[pairId];

            array<float, 4> cost;

            for (int i = 0; i < 4; i++) {
                const double& label0 = ((i & 0x1) == 0) ?
                    triangleValues[triI] : candidates[triI];
                const double& label1 = ((i & 0x2) == 0) ?
                    triangleValues[adjTriI] : candidates[adjTriI];
                
                // FIXME
                cost[i] = trianglePairCost[pairId] * fabs(label0 - label1);
                /*
                cost[i] =
                    trianglePairCost[pairId] *
                    min(fabs(label0 - label1) / (10.0 * adjTriValueVariance), 1.0);
                */
            }

            gModelData->setBinaryTerm(pairId, cost[0], cost[1], cost[2], cost[3]);
        }
        
        // Update unary term
        float cost0 = unaryCostFactor *
            triangleValueUnaryCosts[triI] * totalPairCostFactor;
        float cost1 = unaryCostFactor *
            unaryCosts[triI] * totalPairCostFactor;

        // FIXME
        cost0 = 5; cost1 = 5;

        if (triangleValues[triI] < 0) {
            cost0 = 100000000000.0f;
        }

        if (candidates[triI] < 0) {
            cost1 = 100000000000.0f;
        }

        gModelData->setUnaryTerm(triI, cost0, cost1);

    }
    
    // Perform QPBO...

    printf("Creating QPBO...\n");
    gModelData->qpbo = unique_ptr<GModelData::QPBO>(
            new GModelData::QPBO(gModelData->model));
    printf("Done\n");

    printf("Performing inference...\n");
    gModelData->qpbo->infer();
    printf("Done\n");

    vector<bool> optimalVariables;
    vector<size_t> labels;

    gModelData->qpbo->partialOptimality(optimalVariables);
    gModelData->qpbo->arg(labels);

    size_t numChanged = 0;
    size_t numOptimal = 0;

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        if (optimalVariables[triI]) {
            numOptimal++;

            if (labels[triI] == 1) {
                triangleValues[triI] = candidates[triI];
                triangleValueUnaryCosts[triI] = unaryCosts[triI];

                numChanged++;
            }
        }
    }

    float error = gModelData->model.evaluate(labels);

    printf("RESIDUAL = %f\n", error);
    printf("Num optimal = %d of %d\n", numOptimal, triangles.size());

    return numChanged;
}

struct LineError {
    LineError(
            double _x,
            double _y) :
        x(_x),
        y(_y) {
    }

    template <typename T>
        bool operator()(
                const T* const m,
                // const T* const b,
                T* residuals) const {
            // residuals[0] = m[0] * T(x) + b[0] - T(y);
            residuals[0] = m[0] * T(x) - T(y);
            
            return true;
        }

    double x, y;
};

void TriQPBO::fitCandidateValuesLinear(
        vector<double>& depths,
        int maxPointsToUse) {
    // 1 residuals
    // 1 parameters in block 1 (slope)
    // 1 parameters in block 2 (offset)
    // typedef ceres::AutoDiffCostFunction<LineError, 1, 1, 1> LineCostFunction;
    typedef ceres::AutoDiffCostFunction<LineError, 1, 1> LineCostFunction;

    vector<unique_ptr<LineCostFunction>> costFunctions;

    vector<double> mEstimates;

    for (size_t i = 0; i < points.size(); i++) {
        if (depths[i] > 0 && isfinite(depths[i])) {
            for (const double& existingValue : vertexCandidates[i]) {
                LineCostFunction* costFunction =
                    new LineCostFunction(new LineError(depths[i], existingValue));

                costFunctions.push_back(unique_ptr<LineCostFunction>(costFunction));

                mEstimates.push_back(existingValue / depths[i]);

                if (costFunctions.size() > maxPointsToUse) {
                    break;
                }
            }
        }
    }

    if (costFunctions.size() == 0) {
        return;
    }

    /*
    // Perform RANSAC by randomly selecting cost function terms, solving for
    // model parameters, and counting inliers from the resulting fit.

    // Keep track of the currently-optimal score and value
    double optimalCost = numeric_limits<double>::max();
    double optimalM, optimalB;

    ceres::Problem::Options pOptions;
    // This ensures that ceres don't delete our cost functions
    pOptions.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    pOptions.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    std::random_shuffle(mEstimates.begin(), mEstimates.end());

    // Perform several iterations of RANSAC
    // TODO compute this based on acceptable probability of success
    const int max_iters = min(1000, (int) mEstimates.size());

    // FIXME don't hardcode parameter
    unique_ptr<ceres::LossFunction> robustLoss(new ceres::HuberLoss(0.00000001));

    ceres::Problem problem(pOptions);

    double m, b;

    for (const auto& cf : costFunctions) {
        problem.AddResidualBlock(
                cf.get(),
                robustLoss.get(),
                // &m,
                // &b);
                &m);
    }

    ceres::Solver::Options options;
    // Use default solver
    // options.linear_solver_type = ???;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;

    for (int iter = 0; iter < max_iters; iter++) {
        m = mEstimates[iter];
        // We should expect the linear offset to be 0, so always start with that
        b = 0;

        ceres::Problem::EvaluateOptions options;

        double cost;

        problem.Evaluate(options, &cost, nullptr, nullptr, nullptr);

        if (cost < optimalCost) {
            optimalCost = cost;
            optimalM = m;
            optimalB = b;
        }
    }


    m = optimalM;
    b = optimalB;

    ceres::Solve(options, &problem, &summary);
    */
    sort(mEstimates.begin(), mEstimates.end());
    double m = mEstimates[mEstimates.size() / 2];
    double b = 0;

    for (double& d : depths) {
        d = d * m + b;
    }
}

void TriQPBO::initTriangles() {
    voronoi_diagram<double> vd;

    construct_voronoi(points.begin(), points.end(), &vd);

    vector<size_t> connectedPoints;

    // Loop over all vertices of the voronoi diagram, and all of the edges
    // radiating from each vertex.  The cells associated with each of these
    // edges are the cells which should be connected by a triangle.
    // If there are more than 3 such edges, two triangles are necessary.
    for (auto it = vd.vertices().begin(); it != vd.vertices().end(); ++it) {
        const auto& v = *it;

        connectedPoints.clear();

        const auto* edge = v.incident_edge();

        do {
            if (edge->is_primary()) {
                size_t pointIdx = edge->cell()->source_index();

                connectedPoints.push_back(pointIdx);
            }

            edge = edge->rot_next();

        } while (edge != v.incident_edge());

        // Iterate over each set of three consecutive points, with overlap
        // Note that this essentially constructs a triangle-fan from
        // the connected points.
        for (int i = 0; i < connectedPoints.size() - 1; i+=2) {
            size_t a = connectedPoints[i];
            size_t b = connectedPoints[(i + 1) % connectedPoints.size()];
            size_t c = connectedPoints[(i + 2) % connectedPoints.size()];

            triangles.push_back({a, b, c});
        }
    }
}

void TriQPBO::initTriangleLabels() {
    triangleValues.resize(triangles.size());
    triangleValueUnaryCosts.resize(triangles.size());

    fill(triangleValues.begin(), triangleValues.end(), -1.0);
    fill(triangleValueUnaryCosts.begin(), triangleValueUnaryCosts.end(), 
            std::numeric_limits<float>::max());

    vector<double> candidates;

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const array<size_t, 3>& tri = triangles[triI];

        candidates.clear();

        for (const size_t& vI : tri) {
            for (const auto& c : vertexCandidates[vI]) {
                if (c > 0) {
                    candidates.push_back(c);
                }
            }
        }

        if (candidates.size() == 0) {
            continue;
        }

        std::nth_element(
                &(candidates.front()),
                &(candidates[candidates.size() / 2]),
                &(candidates.back()));

        const double& label = candidates[candidates.size() / 2];

        triangleValues[triI] = label;
    }
}

void TriQPBO::initTriangleColorStats() {
    triangleLabMeanVar.resize(triangles.size());

    array<vector<float>, 3> pixValues;

    for (int i = 0; i < triangles.size(); i++) {
        const auto& tri = triangles[i];

        for (vector<float>& pixels : pixValues) {
            pixels.clear();
        }

        // The following based on the draw_triangle method from CImg.h
        // Note that the gouroud shading and color interpolation implementations
        // in CImg only work on 8-bit integers.
        const double x0 = points[get<0>(tri)].x();
        const double y0 = points[get<0>(tri)].y();
        const double x1 = points[get<1>(tri)].x();
        const double y1 = points[get<1>(tri)].y();
        const double x2 = points[get<2>(tri)].x();
        const double y2 = points[get<2>(tri)].y();
        const double& v = triangleValues[i];

        int nx0 = x0, ny0 = y0, nx1 = x1, ny1 = y1, nx2 = x2, ny2 = y2;
        if (ny0>ny1) cimg::swap(nx0,nx1,ny0,ny1);
        if (ny0>ny2) cimg::swap(nx0,nx2,ny0,ny2);
        if (ny1>ny2) cimg::swap(nx1,nx2,ny1,ny2);

        cimg_forC(imgLab, c) {
            if (ny0<imgLab.height() && ny2>=0) {
                _cimg_for_triangle1(imgLab,xl,xr,y,nx0,ny0,nx1,ny1,nx2,ny2) {
                    int xleft = xl;
                    int xright = xr;

                    if (xright < xleft) {
                        swap(xleft, xright);
                    }

                    for (int x = max(xleft, 0);
                            x <= min(xright, imgLab.width() - 1);
                            x++) {
                        pixValues[c].push_back(imgLab(x, y, 0, c));
                    }
                }
            }
        }

        for (int c = 0; c < 3; c++) {
            const vector<float>& pixels = pixValues[c];
            const CImg<float> pixelsCImg(pixels.data(), pixels.size(), 1, 1, 1,
                    true);

            float mean;
            float var;

            var = pixelsCImg.variance_mean(1, mean);
            
            get<0>(triangleLabMeanVar[i])[c] = mean;
            get<1>(triangleLabMeanVar[i])[c] = var;
        }
    }
}

void TriQPBO::initAdjacency() {
    adjTriCount = 0;

    adjacency.clear();

    adjacency.resize(triangles.size());

    // temporary map from edges (a, b) with a < b to a triangle with that edge
    map<tuple<size_t, size_t>, size_t> edgeTriMap;

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const array<size_t, 3>& tri = triangles[triI];

        for (int i = 0; i < 3; i++) {
            size_t a = tri[i];
            size_t b = tri[(i + 1) % 3];

            if (b < a) {
                swap(a, b);
            }

            const auto edge = make_tuple(a, b);

            if (edgeTriMap.count(edge) == 0) {
                edgeTriMap[edge] = triI;
            } else {
                size_t triA = triI;
                size_t triB = edgeTriMap[edge];

                if (triB < triA) {
                    swap(triA, triB);
                }

                if (adjacency[triA].count(triB) == 0) {
                    adjacency[triA][triB] = {(size_t) adjTriCount, a, b};

                    adjTriCount++;
                }
            }
        }
    }
}

void TriQPBO::initTrianglePairCost() {
    trianglePairCost.resize(adjTriCount);

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        for (auto& pair : adjacency[triI]) {
            size_t adjTriI = pair.first;

            size_t edgeId = pair.second.id;

            const Eigen::Vector3f& u0 = get<0>(triangleLabMeanVar[triI]);
            const Eigen::Vector3f& sigma0 = get<1>(triangleLabMeanVar[triI]);

            const Eigen::Vector3f& u1 = get<0>(triangleLabMeanVar[adjTriI]);
            const Eigen::Vector3f& sigma1 = get<1>(triangleLabMeanVar[adjTriI]);

            // See wikipedia for Bhattacharyya distance of multivariate normal
            // distributions.

            auto sigma = Eigen::DiagonalMatrix<float, 3>((sigma0 + sigma1) * 0.5f);
            auto sigmaInv = sigma.inverse();

            float bDist = (1.0/8.0) * (u0 - u1).transpose() * sigmaInv * (u0 - u1) + 0.5 * log(sigma.diagonal().prod() / sqrt(sigma0.prod() * sigma1.prod()));

            if (!isfinite(bDist)) {
                bDist = 0.0;
            }

            trianglePairCost[edgeId] = bDist;
        }
    }
    
    float var = 0;

    for (float c : trianglePairCost) {
        var += c * c;
    }

    var /= trianglePairCost.size() - 1;

    for (float& c : trianglePairCost) {
        c = exp(-(c * c) / (var));
    }

    // Tabulate the difference in median depths for adjacent triangles
    // with above-median pairwise cost.
    vector<double> adjDepthDiff;

    for (size_t triI = 0; triI < triangles.size(); triI++) {

        const double& triLabel = triangleValues[triI];

        for (auto& pair : adjacency[triI]) {
            size_t adjTriI = pair.first;

            const double& adjTriLabel = triangleValues[adjTriI];

            if (triLabel != adjTriLabel) {
                adjDepthDiff.push_back(fabs(triLabel - adjTriLabel));
            }
        }
    }

    printf("number of adjacent depth samples = %d\n", adjDepthDiff.size());

    std::nth_element(
            &(adjDepthDiff.front()),
            &(adjDepthDiff[adjDepthDiff.size() / 2]),
            &(adjDepthDiff.back()));

    adjTriValueVariance = 0;

    for (int i = 0; i < adjDepthDiff.size() / 2; i++) {
        adjTriValueVariance += adjDepthDiff[i];
    }

    printf("Adjacent triangle value variance = %f\n", adjTriValueVariance);

    adjTriValueVariance /= (adjDepthDiff.size() / 2);

    printf("Adjacent triangle value variance = %f\n", adjTriValueVariance);
}

void TriQPBO::initGModel() {
    gModelData = new GModelData();

    GModelData::Space space(triangles.size(), 2);

    gModelData->model = GModelData::GModel(space);

    // Allocate OpenGM functions for unary and binary terms
    gModelData->unaryFIds.resize(triangles.size());
    gModelData->binaryFIds.resize(adjTriCount);
    
    for (size_t triI = 0; triI < triangles.size(); triI++) {
        // Allocate a unary function for the data term...
        {
            // 2 labels for a single parameter
            size_t shape[] = {(size_t) 2};

            size_t vars[] = {triI};

            GModelData::ExplicitFunction dataTerm(begin(shape), end(shape));

            auto fid = gModelData->model.addFunction(dataTerm);

            gModelData->model.addFactor(fid, begin(vars), end(vars));

            gModelData->unaryFIds[triI] = fid;
        }
        
        // Allocate a binary function for each binary term
        for (auto& pair : adjacency[triI]) {
            size_t adjTriI = pair.first;

            size_t edgeId = pair.second.id;

            // 2 labels for each parameter
            size_t shape[] = {2, 2};

            size_t vars[] = {triI, adjTriI};

            GModelData::ExplicitFunction binTerm(begin(shape), end(shape));

            auto fid = gModelData->model.addFunction(binTerm);

            gModelData->model.addFactor(fid, begin(vars), end(vars));

            gModelData->binaryFIds[edgeId] = fid;
        }
    }
}

