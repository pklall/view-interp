#pragma once

#include "common.h"

#include "localexpansion.hpp"

class Segment;
class Connectivity;
class Segmentation;
class PlanarDepth;

typedef unsigned int imageI_t;

// A handle to a segment
typedef uint16_t segmentH_t;

// A handle to a plane
typedef uint16_t planeH_t;

class Segment {
    private:
        imageI_t minX, maxX;

        imageI_t minY, maxY;

        vector<tuple<imageI_t, imageI_t>> pixels;

        float totalLab[3];

    public:
        Segment();

        inline void addPixel(
                imageI_t x,
                imageI_t y,
                const float lab[3]) {
            minX = min(minX, x);
            maxX = max(maxX, x);
            minY = min(minY, y);
            maxY = max(maxY, y);

            for (int i = 0; i < 3; i++) {
                totalLab[i] += lab[i];
            }

            pixels.push_back(make_tuple(x, y));
        }
        
        inline void avgLab(
                float lab[3]) const {
            for (int i = 0; i < 3; i++) {
                lab[i] = totalLab[i] / size();
            }
        }

        inline void compress() {
            pixels.shrink_to_fit();
        }

        inline unsigned int size() const {
            return pixels.size();
        }

        inline const vector<tuple<imageI_t, imageI_t>>& getPixels() const {
            return pixels;
        }

        inline void getBounds(
                int& _minX,
                int& _minY, 
                int& _maxX,
                int& _maxY) const {
            _minX = minX;
            _minY = minY;
            _maxX = maxX;
            _maxY = maxY;
        }

        inline void getCenter(
                int& x,
                int& y) const {
            x = (minX + maxX) / 2;
            y = (minY + maxY) / 2;
        }
};

class Connectivity {
    friend Segmentation;
    
    private:
        map<segmentH_t, map<segmentH_t, int>> connectivity;

        void increment(
                segmentH_t a,
                segmentH_t b);

    public:
        int getConnectivity(
                segmentH_t a,
                segmentH_t b) const;

        

        inline void forEachNeighbor(
                segmentH_t  s,
                function<void(segmentH_t, int)> fun) const {
            const auto& foundS = connectivity.find(s);

            if (foundS != connectivity.end()) {
                for (const auto& edge : (*foundS).second) {
                    if (edge.second > 0) {
                        fun(edge.first, edge.second);
                    }
                }
            }
        }
};

class Segmentation {
    private:
        vector<Segment> superpixels;

        CImg<segmentH_t> segmentMap;

    public:
        inline const vector<Segment>& getSegments() const {
            return superpixels;
        }

        inline const CImg<segmentH_t>& getSegmentMap() const {
            return segmentMap;
        }

        inline unsigned int size() const {
            return superpixels.size();
        }

        inline const Segment& operator[](segmentH_t index) const {
            return superpixels[index];
        }

        inline segmentH_t& operator()(imageI_t x, imageI_t y) {
            return segmentMap(x, y);
        }

        inline segmentH_t operator()(imageI_t x, imageI_t y) const {
            return segmentMap(x, y);
        }

        void recomputeSegmentMap();

        void createSlicSuperpixels(
                const CImg<float>& lab,
                int numSegments,
                int nc);

        void renderVisualization(
                CImg<float>& result) const;

        void getConnectivity(
                Connectivity& c) const;
};

struct Plane {
    constexpr static const float INVALID = std::numeric_limits<float>::max();

    float cx, cy, c;

    Plane() {
        cx = INVALID;
        cy = INVALID;
        c = INVALID;
    }

    Plane(
            float _cx,
            float _cy,
            float _c) : cx(_cx), cy(_cy), c(_c) {
    }

    inline float dispAt(
            float x,
            float y) const {
        return c + cx * x + cy * y;
    }

    inline bool isValid() const {
        return cx != INVALID && cy != INVALID && c != INVALID;
    }
};

struct StereoProblem {
    CImg<uint16_t> left, right;

    int minDisp;

    int maxDisp;

    CImg<float> disp;

    StereoProblem(
            CImg<uint16_t> left, 
            CImg<uint16_t> right, 
            int minDisp,
            int maxDisp,
            CImg<float> disp);

    inline bool isValidDisp(
            imageI_t x,
            imageI_t y) const {
        return disp(x, y) >= minDisp && disp(x, y) <= maxDisp;
    }
};

class PlanarDepth {
    private:
        const StereoProblem* stereo;
        
        const Segmentation* segmentation;

        vector<Plane> planes;

        vector<planeH_t> segmentPlaneMap;

        bool tabulateSlantSamples(
                const map<imageI_t, vector<tuple<imageI_t, float>>>& dSamples,
                CImg<float>& dtSamples) const;

        bool estimateSlant(
                const map<imageI_t, vector<tuple<imageI_t, float>>>& dSamples,
                float& result) const;

    public:
        PlanarDepth(
                const StereoProblem* _stereo,
                const Segmentation* _segmentation);

        inline const vector<Plane>& getPlanes() const {
            return planes;
        }

        inline const Plane& getPlane(
                segmentH_t segI) const {
            return planes[segmentPlaneMap[segI]];
        }

        inline vector<planeH_t>& getSegmentPlaneMap() {
            return segmentPlaneMap;
        }

        inline const vector<planeH_t>& getSegmentPlaneMap() const {
            return segmentPlaneMap;
        }

        void fitPlanesMedian();

        float getPlaneCostL1(
                segmentH_t segment,
                const Plane& plane) const;

        void getDisparity(
                CImg<float>& disp) const;

        void renderInterpolated(
                float t,
                CImg<float>& result);
};

class PlanarDepthSmoothingProblem {
    private:
        typedef LocalExpansion<segmentH_t, planeH_t> Solver;

        PlanarDepth* depth;

        const Segmentation* segmentation;

        const Connectivity* connectivity;

        unique_ptr<Solver> model;

        float binaryCost(
                segmentH_t a,
                segmentH_t b,
                planeH_t a_label,
                planeH_t b_label);

        float unaryCost(
                segmentH_t n,
                planeH_t n_label);

        void neighborhoodGenerator(
                segmentH_t s,
                vector<segmentH_t>& neighborhood);

        void createModel();

    public:
        float smoothnessCoeff;

        PlanarDepthSmoothingProblem(
                PlanarDepth* _depth,
                const Segmentation* _segmentation,
                const Connectivity* _connectivity);

        void solve();
};

