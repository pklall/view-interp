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
        void initDelaunay();

        void initGModel();

        // Defining this in the header would require pulling in all of
        // opengm, which dramatically slows compilation.
        struct GModelData;

        GModelData* gModelData;

        const CImg<uint8_t>& imgLab;

        vector<Eigen::Vector2f> points;

        /**
         * Adjacency list of edges resulting from delaunay triangulation,
         * annotated with the index of the edge.
         *
         * To avoid counting each edge twice, adjacency[a][b] will exist iff
         * a < b.
         */
        vector<map<size_t, size_t>> adjacency;
        size_t edgeCount;

        /**
         * Duplicate of adjacency for convenience.  Triangle vertices are
         * stored in CCW order.
         */
        vector<array<size_t, 3>> triangles;

        /**
         * The values to merge.  Negative values are ignored.
         */
        vector<double> existingValue;
        vector<double> newValue;
};

