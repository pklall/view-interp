#include "rectification.h"

StereoTransform::StereoTransform(int numCoeffs, float centerX, float centerY) {
    homography.identity_matrix(4);

    center[0] = centerX;
    center[1] = centerY;
    
    radial = std::vector<float>(numCoeffs, 0.0f);
    tangential = std::vector<float>(numCoeffs, 0.0f);
}

void StereoTransform::apply(
        float in[2],
        float out[2]) {
    // See http://en.wikipedia.org/wiki/Distortion_(optics)#Software_correction
    
    float r2 = sqr(in[0] - center[0]) + sqr(in[1] - center[1]);

    for (int d = 0; d < 2; d++) {
        float radTotal = 1;
        float coeff = 1;
        for (float K : radial) {
            coeff *= r2;
            radTotal += K * coeff;
        }

        int dp = (d + 1) % 2;


        float tangSeries = 1;
        coeff = 1;
        for (int tInd = 2; tInd < (int) tangential.size(); tInd++) {
            coeff *= r2;
            tangSeries += tangential[tInd] * coeff;
        }

        float tangTotal = tangential[d] * (r2 + 2 * sqr(in[d] - center[d]));
        tangTotal += 2 * tangential[dp] * (in[d] - center[d]) * (in[dp] - center[dp]);
        tangTotal *= tangSeries;

        out[d] = (in[d] - center[d]) * radTotal + tangTotal;
    }
}

void StereoTransform::apply(
        const CImg<float>& in,
        CImg<float>& out) {
}
