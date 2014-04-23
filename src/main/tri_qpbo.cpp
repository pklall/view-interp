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

    initTriangles();

    initTriangleColorStats();

    initAdjacency();

    initGModel();

    fill(triangleValues.begin(), triangleValues.end(), -1.0);
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

    for (int triI = 0; triI < triangles.size(); triI++) {
        const auto& tri = triangles[triI];

        for (int c = 0; c < 3; c++) {
            color[c] = triangleAvgLab[triI][c];
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

    // Draw adjacency graph as lines between the centers of triangles
    float black[3] = {0, 0, 0};

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

            colorVis.draw_line(cx, cy, cx2, cy2, black);
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

    for (size_t i = 0; i < depths.size(); i++) {
        if (depths[i] > 0 && isfinite(depths[i])) {
            vertexCandidates[i].push_back(depths[i]);
        }
    }
}

void TriQPBO::addCandidateVertexDepths(
        const vector<double>& depths) {
    assert(depths.size() == points.size());

    for (size_t i = 0; i < depths.size(); i++) {
        if (depths[i] > 0 && isfinite(depths[i])) {
            vertexCandidates[i].push_back(depths[i]);
        }
    }
}

void TriQPBO::solve() {
    vector<double> candidates;

    for (size_t triI = 0; triI < triangles.size(); triI++) {
        const array<size_t, 3>& tri = triangles[triI];

        candidates.clear();

        for (const size_t& vI : tri) {
            candidates.insert(candidates.end(),
                    vertexCandidates[vI].begin(),
                    vertexCandidates[vI].end());
        }

        // FIXME
        sort(candidates.begin(), candidates.end());

        if (candidates.size() == 0) {
            triangleValues[triI] = 0;
            continue;
        }

        triangleValues[triI] = candidates[0];
    }

       /*
    qpbo = unique_ptr<QPBO>(new QPBO(model));
    qpbo->infer();

    vector<bool> optimalVariables;
    vector<size_t> labels;

    qpbo->partialOptimality(optimalVariables);

    qpbo->arg(labels);

    int nodeIndex = 0;

    int numChanged = 0;

    for (const node_t& node : nodes) {
        if (optimalVariables[nodeIndex] && labels[nodeIndex] == 1) {
            (*labeling)[node] = candidateGenerator(node);

            numChanged++;
        }

        nodeIndex++;
    }

    return numChanged;
    */
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

    printf("m = %f\n, b = %f\n", m, b);

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

    triangleValues.resize(triangles.size());

    fill(triangleValues.begin(), triangleValues.end(), -1);
}

void TriQPBO::initTriangleColorStats() {
    triangleAvgLab.resize(triangles.size());

    for (int i = 0; i < triangles.size(); i++) {
        const auto& tri = triangles[i];

        triangleAvgLab[i].fill(0);

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

        int numPixels = 0;

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
                        triangleAvgLab[i][c] += (float) imgLab(x, y, 0, c);

                        numPixels++;
                    }
                }
            }
        }

        numPixels /= 3;

        for (int c = 0; c < 3; c++) {
            triangleAvgLab[i][c] /= numPixels;
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

void TriQPBO::initGModel() {
    gModelData = new GModelData();

    GModelData::Space space(triangles.size(), 2);

    gModelData->model = GModelData::GModel(space);

    // Allocate OpenGM functions for unary and binary terms
    gModelData->unaryFIds.resize(triangles.size());
    gModelData->binaryFIds.resize(adjTriCount);
    
    for (size_t triI = 0; triI < triangles.size(); triI++) {
        // Allocate a unary function for the data term...

        // 2 labels for a single parameter
        size_t shape[] = {(size_t) 2};

        size_t vars[] = {triI};

        GModelData::ExplicitFunction dataTerm(begin(shape), end(shape));

        auto fid = gModelData->model.addFunction(dataTerm);

        gModelData->model.addFactor(fid, begin(vars), end(vars));

        gModelData->unaryFIds[triI] = fid;
        
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

