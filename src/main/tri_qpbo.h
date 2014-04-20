#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <vector>
#include <map>

/**
 * Merges candidate values at sparse samples by triangulating and then
 * fitting planes via QPBO.
 */
class TriQPBO {
    private:
        /*
        struct UnaryCost {
            float operator()(
                    size_t node,
                    float label) {
            }

            const TriQPBO* self;
        };

        struct BinaryCost {
            float operator()(
                    size_t node0,
                    size_t node1,
                    float label0,
                    float label1) {
                return fabs(label0 - label1);
            }

            const TriQPBO* self;
        };
 
        typedef LocalExpansion<size_t, float, UnaryCost, BinaryCost>
            QPBO;
        */

    public:
        void init(
                const CImg<uint8_t>& lab,
                const vector<Eigen::Vector2f>& points,
                const vector<float>& initDepth);

        void visualizeTriangulation(
                CImg<uint8_t>& colorVis);

        void merge(
                const vector<float>& newDepth);

    private:
        vector<Eigen::Vector2f> points;

        /**
         * Adjacency list of edges resulting from delaunay triangulation,
         * annotated with the index of the edge.
         *
         * To avoid counting each edge twice, adjacency[a][b] will exist iff
         * a < b.
         */
        vector<map<size_t, size_t>> adjacency;

        vector<float> edgeWeights;

        /**
         * Duplicate of adjacency for convenience.  Triangle vertices are
         * stored in CCW order.
         */
        vector<tuple<size_t, size_t, size_t>> triangles;

        vector<float> depth;
};

