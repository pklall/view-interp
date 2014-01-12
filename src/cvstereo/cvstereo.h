#pragma once

#include "common.h"

class CVStereo {
    private:
        cv::Mat original[2];

        cv::Mat rectTransforms[2];

        cv::Mat rectified[2];

        cv::Mat rectifiedMasks[2];

        int minDisparity;

        int numDisparities;

        cv::Mat stereoDisparity;

        void convertCImgToMat(
                const CImg<float>& in,
                cv::Mat& out);

        void convertMatToCImg(
                const cv::Mat& in,
                CImg<float>& out);

        void warp(std::vector<cv::Point2f> points[2]);

        void rectify();

        void processPrerectified();

    public:
        CVStereo(
                const CImg<float>& left,
                const CImg<float>& right,
                bool prerectified = false);

        void getRectified(
                CImg<float>& left,
                CImg<float>& right);

        void matchStereo(
                int minDisparity = -256,
                int maxDisparity = 256,
                int windowSize = 3,
                float smoothnessScale = 1.0f);

        void getStereo(CImg<float>& out);
};

