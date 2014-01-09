#pragma once

#include "common.h"

/**
 * Transformation for performing stereo rectification including
 * both a homography as well as Brown's distortion model.
 */
class StereoTransform {
    private:
        CImg<float> homography;
        
        float center[2];
        std::vector<float> radial;
        std::vector<float> tangential;

    public:
        /**
         * Constructor
         *
         * \param numCoeffs The number of radial and tangential coefficients
         *        (each) to use
         * \param centerX The center of lens distortion
         * \param centerY The center of lens distortion
         */
        StereoTransform(
                int numCoeffs,
                float centerX,
                float centerY);

        void apply(
                float in[2],
                float out[2]);

        void apply(
                const CImg<float>& in,
                CImg<float>& out);

        void fillParamVector(
                std::vector<float*>& params);
};

