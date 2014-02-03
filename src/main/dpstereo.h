#pragma once

#include "common.h"

#include "segment.h"

class DPStereo {
    private:
        const Segmentation* segmentation;

        int smallDisp;

        float costLargeDispSegEdge;

        float costLargeDisp;

    public:
        DPStereo(
                const Segmentation* _segmentation,
                int _smallDisp,
                float _costLargeDispSegEdge,
                float _costLargeDisp);

        void computeStereo(
                StereoProblem& problem);
};
