#pragma once

#include "common.h"

namespace ogm = opengm;

typedef ogm::SimpleDiscreteSpace<> Space;

typedef ogm::meta::TypeListGenerator<
    ogm::ExplicitFunction<float>,
    ogm::PottsFunction<float>
>::type FunctionTypeList;

typedef ogm::GraphicalModel<float, ogm::Adder, FunctionTypeList, Space> GModel;

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

class AdaptBPStereo {
    private:
        CImg<int16_t> left, right;

        int minDisp;

        int maxDisp;

        float omega;

        float smoothFactor;

        int numSuperpixels;

        vector<vector<tuple<uint16_t, uint16_t>>> superpixels;

        CImg<int> segmentation;

        CImg<int16_t> disp;

        vector<Plane> planes;

        CImg<float> segmentPlaneCost;

        GModel mrf;

        vector<size_t> superpixelPlaneMap;

        void computeGreedyDisp();

        bool estimateSlant(
            const map<uint16_t, vector<tuple<uint16_t, int16_t>>>& dSamples,
            float& result);

        void fitPlanes();

        void computeSegmentPlaneCost();

        void mergeSegmentsByPlane();

        void computeGreedySuperpixelPlaneMap();

        void createMRF();

        void solveMRF();

    public:
        AdaptBPStereo(
                const CImg<int16_t>& left,
                const CImg<int16_t>& right,
                int minDisp,
                int maxDisp);

        void computeStereo();

        void getDisparity(
                CImg<float>& disp
                );
};
