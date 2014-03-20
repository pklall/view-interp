#pragma once

#include <Eigen/Dense>

#include "common.h"

#include "polar_rectification.h"

#include "segment.h"

class PolarStereo {
    public:
        PolarStereo(
                int maxRectificationPixels);

        void computeStereo(
                const CImg<uint8_t>& leftLab,
                const CImg<uint8_t>& rightLab,
                const Segmentation* segmentation,
                int minDisp,
                int maxDisp,
                Eigen::Matrix3d F,
                array<Eigen::Vector2d, 2> match);
        
    private:
        CImgList<float> disparityPyramid;

        PolarRectification rectifier;
        
        // Buffer for storing rectified segments
        unique_ptr<uint8_t[]> rectificationBuffer;
};
