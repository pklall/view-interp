#include "tri_qpbo.h"

#include "boost/polygon/voronoi.hpp"

using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;

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

void TriQPBO::init(
        const CImg<uint8_t>& lab,
        const vector<Eigen::Vector2f>& _points,
        const vector<float>& _depth) {
    depth = _depth;
    points = _points;

    edgeWeights.clear();

    adjacency.clear();

    adjacency.resize(points.size());

    voronoi_diagram<double> vd;

    construct_voronoi(points.begin(), points.end(), &vd);

    printf("num cells = %d\n\n", vd.num_cells());

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

            triangles.push_back(make_tuple(a, b, c));

            size_t pairs[3][2] = {
                {a, b},
                {b, c},
                {c, a}
            };

            for (int p = 0; p < 3; p++) {
                size_t i0 = min(pairs[p][0], pairs[p][1]);
                size_t i1 = max(pairs[p][0], pairs[p][1]);

                if (adjacency[i0].count(i1) == 0) {
                    adjacency[i0][i1] = edgeWeights.size();
                    edgeWeights.push_back(0);
                }
            }
        }
    }

    printf("Triangle count = %d\n", triangles.size());
}

void TriQPBO::visualizeTriangulation(
        CImg<uint8_t>& colorVis) {
    uint8_t color[3];

    const CImg<uint8_t>& lines = CImg<uint8_t>::lines_LUT256();

    for (int i = 0; i < triangles.size(); i++) {
        const auto& tri = triangles[i];

        cimg_forC(lines, c) {
            color[c] = lines(i % 256, 0, 0, c);
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
}

void TriQPBO::merge(
        const vector<float>& newDepth) {
}
