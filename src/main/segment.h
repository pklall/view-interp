#pragma once

#include "common.h"

class Superpixel {
    private:
        uint16_t minX, maxX;

        uint16_t minY, maxY;

        vector<tuple<uint16_t, uint16_t>> pixels;

    public:
        inline void addPixel(
                uint16_t x,
                uint16_t y) {
            pixels.push_back(make_tuple(x, y));

            if (pixels.size() == 0) {
                minX = x;
                maxX = x;
                minY = y;
                maxY = y;
            } else {
                minX = min(minX, x);
                maxX = max(maxX, x);
                minY = min(minY, y);
                maxY = max(maxY, y);
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
                int& minX,
                int& minY, 
                int& maxX,
                int& maxY) const {
            minX = this->minX;
            minY = this->minY;
            maxX = this->maxX;
            maxY = this->maxY;
        }
};

class Segmentation {
    private:
        vector<Superpixel> superpixels;

        CImg<int> segmentMap;

    public:
        vector<Superpixel>& getSuperpixels() {
            return superpixels;
        }

        const vector<Superpixel>& getSuperpixels() const {
            return superpixels;
        }

        inline const CImg<int>& getSegmentMap() const {
            return segmentMap;
        }

        inline size_t size() const {
            return superpixels.size();
        }

        inline Superpixel& operator[](size_t index) {
            return superpixels[index];
        }

        inline const Superpixel& operator[](size_t index) const {
            return superpixels[index];
        }

        inline int& operator()(size_t x, size_t y) {
            return segmentMap(x, y);
        }

        inline int operator()(size_t x, size_t y) const {
            return segmentMap(x, y);
        }

        void recomputeSegmentMap();

        void createSlicSuperpixels(
                const CImg<float>& lab,
                int numSuperpixels,
                int nc);

        void renderVisualization(
                CImg<float>& result);
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

    inline bool isValidDisp(
            size_t x,
            size_t y) const {
        return disp(x, y) >= minDisp && disp(x, y) <= maxDisp;
    }
};

class PlanarDepth {
    private:
        const StereoProblem& stereo;
        
        const Segmentation& segmentation;

        vector<Plane> planes;

        bool estimateSlant(
                const map<uint16_t, vector<tuple<uint16_t, float>>>& dSamples,
                float& result) const;

        void fitPlanes();

    public:
        PlanarDepth(
                const StereoProblem& _stereo,
                const Segmentation& _segmentation);

        void getDisparity(
                CImg<float>& disp) const;
};
