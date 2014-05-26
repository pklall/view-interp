#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <vector>
#include <list>
#include <set>
#include <map>

/**
 * Merges candidate values at sparse samples by triangulating and then
 * running QPBO MRF-minimization on the resulting graph.
 */
class TriQPBO {
    public:
        TriQPBO(
                const CImg<float>& lab,
                const vector<Eigen::Vector2f>& points);

        void init();

        ~TriQPBO();

        void denseInterp(
                CImg<double>& result);

        void visualizeTriangulation(
                CImg<float>& colorVis);

        /**
         * Note that this may modify depths in-place if fitLinear is true!
         */
        void addCandidateVertexDepths(
                vector<double>& depths,
                bool fitLinear);

        void addCandidateVertexDepths(
                const vector<double>& depths);

        void solveAlphaExpansion(
                double minDepth,
                double maxDepth,
                int numLabels,
                int numIters,
                float unaryCostFactor);

        void solve(
                int numIters,
                float unaryCostFactor);

        void solveSmooth();

        void solveSmoothHack();

        void smoothAvg();

        inline void getSmoothTriangles(
                vector<array<Eigen::Vector3d, 3>>& result) {
            for (size_t triI = 0; triI < triangles.size(); triI++) {
                const array<size_t, 3>& tri = triangles[triI];

                array<Eigen::Vector3d, 3> vertices;

                for (int vI = 0; vI < 3; vI++) {
                    vertices[vI] = Eigen::Vector3d(
                            points[tri[vI]].x(),
                            points[tri[vI]].y(),
                            smoothTriangleValues[triI][vI]);
                }

                result.push_back(vertices);
            }
        }

    private:
        template<typename T>
        inline T unaryCost(
                size_t vertexIndex,
                T candidateValue) const {
            T cost = T(0);

            for (const double& c : vertexCandidates[vertexIndex]) {
                T val = T(c);

                // cost += min(
                        // abs(candidateValue - val) /
                        // T(adjTriValueVariance * 5.0), 1.0);

                T c1 = candidateValue - val;
                cost += c1 * c1;
            }

            cost /= T(vertexCandidates.size());

            return cost;
        }

        template<typename T>
        inline T unaryCostTri(
                size_t triIndex,
                T candidateValue) const {
            T cost = T(0);

            for (const size_t& vI : triangles[triIndex]) {
                cost += unaryCost(vI, candidateValue);
            }

            cost /= T(3.0);

            return cost;
        }

        /**
         * Uses QPBO to fuse the candidate labels with the existing triangle
         * labeling.
         *
         * Returns the number of modified labels.
         */
        size_t mergeCandidateValues(
                const vector<double>& candidates,
                const vector<float>& candidateUnaryCosts,
                float unaryCostFactor);

        /**
         * Uses a RANSAC-like procedure to robustly solve for a linear (plus
         * offset) relationship between candidate and current values.
         *
         * This can be used to merge new observations computed with an
         * unknown scale.
         */
        void fitCandidateValuesLinear(
                vector<double>& depths,
                int maxPointsToUse = 20000);

        void initTriangles();

        void initTriangleLabels();

        void initTriangleColorStats();

        void initAdjacency();

        void initTrianglePairCost();

        void initGModel();

        // Definining this in the cpp file prevents pulling all of opengm's
        // massive headers into this one.
        struct GModelData;

        struct TriAdj {
            size_t id;
            size_t edgePt0;
            size_t edgePt1;
        };

        GModelData* gModelData;

        const CImg<float> imgLab;

        const vector<Eigen::Vector2f> points;

        /**
         * The vertex values to merge.  Negative values are considered invalid.
         */
        vector<vector<double>> vertexCandidates;

        /**
         * Triangle vertices are stored as indices into `points` in CCW order.
         */
        vector<array<size_t, 3>> triangles;
        
        /**
         * Average Lab color among all pixels in each triangle.
         */
        vector<tuple<Eigen::Vector3f, Eigen::Vector3f>> triangleLabMeanVar;

        /**
         * Adjacency list representation of adjacent triangles.
         * adjacency[a] = {(b_0, idx_0), ..., (b_n, idx_n)}
         * for pairs of adjacent triangles (a, b) and edge indexed by idx.
         *
         * To avoid counting each edge twice, adjacency[a][b] will exist iff
         * a < b.
         */
        vector<map<size_t, TriAdj>> adjacency;

        /**
         * The number of adjacent triangles
         */
        size_t adjTriCount;

        /**
         * The estimated variance in value between similar adjacent triangles.
         */
        double adjTriValueVariance;

        /**
         * Pairwise energy term coefficient between adjacent triangles.
         */
        vector<float> trianglePairCost;

        /**
         * The current triangle labels.
         */
        vector<double> triangleValues;
        
        /**
         * The unary cost associated with each current triangle label.
         */
        vector<float> triangleValueUnaryCosts;

        vector<array<double, 3>> smoothTriangleValues;
};

