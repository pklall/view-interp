#pragma once

#include "common.h"

#include "segment.h"

namespace ogm = opengm;

typedef ogm::SimpleDiscreteSpace<> Space;

typedef ogm::meta::TypeListGenerator<
    ogm::ExplicitFunction<float>,
    ogm::PottsFunction<float>
>::type FunctionTypeList;

typedef ogm::GraphicalModel<float, ogm::Adder, FunctionTypeList, Space> GModel;

class AdaptBPStereo {
    private:
        CImg<int16_t> left, right;

        int minDisp;

        int maxDisp;

        float omega;

        float smoothFactor;

        int numSuperpixels;

        Segmentation segmentation;

        CImg<float> disp;

        vector<Plane> planes;

        CImg<float> segmentPlaneCost;

        GModel mrf;

        vector<size_t> superpixelPlaneMap;

        void computeGreedyDisp();

        bool estimateSlant(
            const map<uint16_t, vector<tuple<uint16_t, float>>>& dSamples,
            float& result);

        void fitPlanes(
            bool assignSegmentsToPlanes);

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
                CImg<float>& disp);
};
