#pragma once

#include "common.h"

/**
 * Computes stereo correspondence based on 
 *
 * Segment-Based Stereo Matching Using Belief Propagation and a Self-Adapting
 * Dissimilarity Measure (by Klause, Sormann, and Karner)
 *
 * a.k.a "AdaptBP" in Middlebury rankings
 *
 */
void computeAdaptBPStereo(
        const CImg<int16_t>& left,
        const CImg<int16_t>& right,
        int minDisp,
        int maxDisp,
        CImg<int16_t>& disp);
