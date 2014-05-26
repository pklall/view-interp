#pragma once

#include "common.h"

void featureMatchCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& matches);

void visualizeCorrespondence(
        const CImg<float>& from,
        const CImg<int>& corr,
        const CImg<float>& to);

void visualizeVectorImg(
        const CImg<float>& vec);

void displayClickable(
        CImg<float>& img,
        std::function<void(int, int)> onClick);

