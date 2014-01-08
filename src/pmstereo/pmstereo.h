#pragma once

#include "common.h"

void patchMatchTranslationalCorrespondence(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        const CImg<float>& grad1,
        const CImg<float>& grad2,
        CImg<float>& fieldLeft,
        CImg<float>& fieldRight,
        CImg<float>& distLeft,
        CImg<float>& distRight,
        CImg<int>& sortedLeft,
        CImg<int>& sortedRight,
        int wndSize,
        int iterations,
        float randomSearchFactor,
        int increment);
