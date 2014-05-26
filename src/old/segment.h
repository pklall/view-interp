#pragma once

#include "common.h"

#include "localexpansion.hpp"

#include "cvutil/cvutil.h"

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

    public:
        Segment();

        inline void addPixel(
                imageI_t x,
                imageI_t y) {
            minX = min(minX, x);
            maxX = max(maxX, x);
            minY = min(minY, y);
            maxY = max(maxY, y);

            pixels.push_back(make_tuple(x, y));
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

        vector<array<float, 3>> medianLab;

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

        inline const Segment& operator[](
                segmentH_t index) const {
            return superpixels[index];
        }

        inline segmentH_t& operator()(
                imageI_t x,
                imageI_t y) {
            return segmentMap(x, y);
        }

        inline segmentH_t operator()(
                imageI_t x,
                imageI_t y) const {
            return segmentMap(x, y);
        }

        inline array<float, 3>& medLab(
                segmentH_t s) {
            return medianLab[s];
        }

        inline const array<float, 3>& medLab(
                segmentH_t s) const {
            return medianLab[s];
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
    CImg<int16_t> left, right;
    CImg<int16_t> leftLab, rightLab;

    int minDisp;

    int maxDisp;

    CImg<float> disp;

    StereoProblem(
            CImg<int16_t> left, 
            CImg<int16_t> right, 
            int minDisp,
            int maxDisp);

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

    private:
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

        void getPlaneCostL1(
                segmentH_t segment,
                const Plane& plane,
                float& cost,
                int& samples) const;

        void getDisparity(
                CImg<float>& disp) const;

        void renderInterpolated(
                float t,
                CImg<float>& result);

        inline bool isInBounds(
                segmentH_t segH,
                const Plane& plane) const {
            if (!plane.isValid()) {
                return false;
            }

            int minX, minY, maxX, maxY;

            int minDisp = stereo->minDisp;
            int maxDisp = stereo->maxDisp;

            (*segmentation)[segH].getBounds(minX, minY, maxX, maxY);

            for (int i = 0; i < 4; i++) {
                int x = ((i & 0x01) == 0) ? minX : maxX;
                int y = ((i & 0x02) == 0) ? minY : maxY;

                float disp = plane.dispAt(x, y);

                int rx = (int) (x - disp + 0.5f);

                if (disp > maxDisp ||
                        disp < minDisp ||
                        rx <= 0 ||
                        rx >= stereo->right.width() - 1) {
                    return false;
                }
            }

            return true;
        }
};

class PlanarDepthSmoothingProblem {
    private:
        struct UnaryCost {
            PlanarDepthSmoothingProblem* self;

            float operator()(
                segmentH_t n,
                planeH_t n_label);
        };

        struct BinaryCost {
            PlanarDepthSmoothingProblem* self;

            float operator()(
                segmentH_t a,
                segmentH_t b,
                planeH_t a_label,
                planeH_t b_label);
        };

        typedef LocalExpansion<segmentH_t, planeH_t, UnaryCost, BinaryCost> Solver;

        // typedef GMM<6> GMM6;

    private:
        const size_t numSegmentsPerExpansion = 30;

        const StereoProblem* stereo;

        const Segmentation* segmentation;

        const Connectivity* connectivity;

        PlanarDepth* depth;

        // unique_ptr<GMM6> edgeModel;
        
        unique_ptr<Solver> model;

        float smoothnessCoeff;

        float unaryInlierThresh;

        float binaryC0InlierThresh;

        float minDisp;

        float maxDisp;

        float meanSlope;

        float sdSlope;

        vector<float> medianColorDiff;

    private:
        inline float pairwiseL1PlaneDist(
                segmentH_t segA,
                segmentH_t segB,
                planeH_t planeA,
                planeH_t planeB) {
            int cX1, cY1;
            int cX2, cY2;

            (*segmentation)[segA].getCenter(cX1, cY1);
            (*segmentation)[segB].getCenter(cX2, cY2);

            // Crude approximation of the point between segments
            // segI and nI.  This enables approximation of the
            // depth discontinuity between segments.
            float middleX = (cX1 + cX2) / 2.0f;
            float middleY = (cY1 + cY2) / 2.0f;

            const Plane& p1 = depth->getPlanes()[planeA];
            const Plane& p2 = depth->getPlanes()[planeB];

            float d1 = p1.dispAt(middleX, middleY);
            float d2 = p2.dispAt(middleX, middleY);

            return fabs(d1 - d2);
        }

        inline float pairwiseColorDist(
                segmentH_t segA,
                segmentH_t segB) {
            const array<float, 3>& labA = (*segmentation).medLab(segA);
            const array<float, 3>& labB = (*segmentation).medLab(segB);

            float dist = 0.0f;

            for (int c = 0; c < 3; c++) {
                dist += fabs(labA[c] - labB[c]);
            }

            return dist;
        }

        inline bool colorConnected(
                segmentH_t segA,
                segmentH_t segB) {
            float colorDiff = pairwiseColorDist(segA, segB);

            if (colorDiff < max(medianColorDiff[segA], medianColorDiff[segB])) {
                return true;
            } else {
                return false;
            }
        }

        void neighborhoodGenerator(
                segmentH_t s,
                vector<segmentH_t>& neighborhood);

        void computeUnaryCostStats();

        void computePairwiseCostStats();

        void createModel();

    public:
        inline void setSmoothness(
                float s) {
            smoothnessCoeff = s;
        }

        PlanarDepthSmoothingProblem(
                PlanarDepth* _depth,
                const StereoProblem* _stereo,
                const Segmentation* _segmentation,
                const Connectivity* _connectivity);

        void computeInlierStats();

        void solve();

        void visualizeUnaryCost(
                CImg<float>& vis);
};

