#pragma once

#include "common.h"

#include "polar_stereo.h"

#include "daisy/daisy.h"

#include <Eigen/Dense>

#include <nanoflann.hpp>

class SparseDaisyStereo {
    public:
        void init(
                const CImg<uint8_t>& mainImgGray);

        void match(
                const CImg<uint8_t>& otherImgGray,
                const PolarFundamentalMatrix& F,
                const vector<Eigen::Vector2f>& samples,
                vector<Eigen::Vector2f>& matches,
                vector<float>& matchDistances);

    private:
        void initializeDaisyForImg(
                const CImg<uint8_t>& img,
                daisy& d);

        unique_ptr<daisy> mainDaisy;
};

