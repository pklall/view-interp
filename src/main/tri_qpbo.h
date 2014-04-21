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
                const CImg<uint8_t>& lab,
                const vector<Eigen::Vector2f>& points,
                const vector<double>& initValue);

        ~TriQPBO();

        void denseInterp(
                CImg<double>& result);

        void visualizeTriangulation(
                CImg<uint8_t>& colorVis);

        inline void setCandidateValue(
                size_t idx,
                double value) {
            newValue[idx] = value;
        }

        inline vector<double>& getCandidateValueArray() {
            return newValue;
        }

        void fitCandidateValuesLinear();

        void computeFusion();

    private:
        void initTriangles();

        void initAdjacency();

        void initGModel();

        // Definining this in the cpp file prevents pulling all of opengm's
        // massive headers into this one.
        struct GModelData;

        GModelData* gModelData;

        const CImg<uint8_t>& imgLab;

        vector<Eigen::Vector2f> points;

        /**
         * Triangle vertices are stored as indices into `points` in CCW order.
         */
        vector<array<size_t, 3>> triangles;

        /**
         * Adjacency list representation of adjacent triangles.
         * adjacency[a] = {(b_0, idx_0), ..., (b_n, idx_n)}
         * for pairs of adjacent triangles (a, b) and edge indexed by idx.
         *
         * To avoid counting each edge twice, adjacency[a][b] will exist iff
         * a < b.
         */
        vector<map<size_t, size_t>> adjacency;

        /**
         * The number of adjacent triangles
         */
        int adjTriCount;


        /**
         * The values to merge.  Negative values are ignored.
         */
        vector<double> existingValue;
        vector<double> newValue;
};

