#pragma once

#include "common.h"

class Superpixel;
class Connectivity;
class Segmentation;
class PlanarDepth;

class Superpixel {
    private:
        uint16_t minX, maxX;

        uint16_t minY, maxY;

        vector<tuple<uint16_t, uint16_t>> pixels;

        float totalLab[3];

    public:
        Superpixel();

        inline void addPixel(
                uint16_t x,
                uint16_t y,
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

        inline size_t size() const {
            return pixels.size();
        }

        inline const vector<tuple<uint16_t, uint16_t>>& getPixels() const {
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
        map<size_t, map<size_t, int>> connectivity;

        void increment(
                size_t a,
                size_t b);

    public:
        int getConnectivity(
                size_t a,
                size_t b) const;

        inline void forEachNeighbor(
                size_t s,
                function<void(size_t, int)> fun) const {
            const auto& foundS = connectivity.find(s);

            if (foundS != connectivity.end()) {
                for (const auto& edge : (*foundS).second) {
                    fun(edge.first, edge.second);
                }
            }
        }
};

class Segmentation {
    private:
        vector<Superpixel> superpixels;

        CImg<size_t> segmentMap;

    public:
        inline const vector<Superpixel>& getSuperpixels() const {
            return superpixels;
        }

        inline const CImg<size_t>& getSegmentMap() const {
            return segmentMap;
        }

        inline size_t size() const {
            return superpixels.size();
        }

        inline const Superpixel& operator[](size_t index) const {
            return superpixels[index];
        }

        inline size_t& operator()(size_t x, size_t y) {
            return segmentMap(x, y);
        }

        inline size_t operator()(size_t x, size_t y) const {
            return segmentMap(x, y);
        }

        void recomputeSegmentMap();

        void createSlicSuperpixels(
                const CImg<float>& lab,
                int numSuperpixels,
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
            size_t x,
            size_t y) const {
        return disp(x, y) >= minDisp && disp(x, y) <= maxDisp;
    }
};

class PlanarDepth {
    private:
        const StereoProblem* stereo;
        
        const Segmentation* segmentation;

        vector<Plane> planes;

        bool estimateSlant(
                const map<uint16_t, vector<tuple<uint16_t, float>>>& dSamples,
                float& result) const;

        void fitPlanes();

    public:
        PlanarDepth(
                const StereoProblem* _stereo,
                const Segmentation* _segmentation);

        void getDisparity(
                CImg<float>& disp) const;

        void renderInterpolated(
                float t,
                CImg<float>& result);
};

class SegmentLabelProblem {
    private:
        typedef opengm::SimpleDiscreteSpace<> Space;

        typedef opengm::meta::TypeListGenerator<
            opengm::ExplicitFunction<float>,
            opengm::PottsFunction<float>
                >::type FunctionTypeList;

        typedef opengm::GraphicalModel<float, opengm::Adder, FunctionTypeList,
                Space> GModel;

        const Segmentation* segmentation;

        size_t numLabels;

        GModel model;

        vector<size_t> labels;

    public:
        SegmentLabelProblem(
                const Segmentation* _segmentation,
                size_t _numLabels);

        void addUnaryFactor(
                size_t segment,
                function<float(size_t)> labelFactor);

        void addBinaryFactor(
                size_t segment1,
                size_t segment2,
                float factor);

        void solveMAP();

        inline const vector<size_t>& getLabels() const {
            return labels;
        }
};

