#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <vector>
#include <map>

/**
 * Merges candidate values at sparse samples by triangulating and then
 * running QPBO MRF-minimization on the resulting graph.
 */
class TriQPBO {
    public:
        TriQPBO(
                const CImg<float>& lab,
                const vector<Eigen::Vector2f>& points,
                const vector<double>& values);

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

        void solve();

    private:
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
        float adjTriValueVariance;

        /**
         * Pairwise energy term coefficient between adjacent triangles.
         */
        vector<float> trianglePairCost;

        /**
         * The current triangle labels.
         */
        vector<double> triangleValues;
};

