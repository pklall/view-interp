#pragma once

#include "common.h"

#include <opencv/cv.h>
#include <opencv/ml.h>

void convertCImgToMat(
        const CImg<float>& in,
        cv::Mat& out);

void convertMatToCImg(
        const cv::Mat& in,
        CImg<float>& out);

void slicSuperpixels(
        const CImg<float>& labIn,
        int numSuperpixels,
        int nc,
        CImg<uint16_t>& result);

class CVFeatureMatcher {
    private:
        vector<cv::KeyPoint> keypoints;

        std::vector<cv::DMatch> cvMatchList;

        cv::Mat descriptors;

        unique_ptr<cv::ORB> orb;

        int normType;
        
    public:
        inline void getKeypoint(
                int index,
                float& x,
                float& y) const {
            x = keypoints[index].pt.x;
            y = keypoints[index].pt.y;
        }

        CVFeatureMatcher(
                int _maxPoints);

        void detectFeatures(
                const CImg<uint8_t>& grayImg);

        void match(
                const CVFeatureMatcher& other,
                vector<tuple<int, int>>& matchList,
                int maxMatches);

        void match(
                const CVFeatureMatcher& other,
                vector<tuple<float, float, float, float>>& matchedPoints);
};

class CVStereo {
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

    private:
        void warp(std::vector<cv::Point2f> points[2]);

        void rectify();

        void processPrerectified();

        cv::Mat original[2];

        cv::Mat rectTransforms[2];

        cv::Mat rectified[2];

        cv::Mat rectifiedMasks[2];

        int minDisparity;

        int numDisparities;

        cv::Mat stereoDisparity;
};

template<int DIM>
class GMM {
    public:
        GMM(
                int numGaussians) {
            em = cv::EM(numGaussians);
        }

        /**
         * Trains the mixture model with the provided samples.
         */
        inline void train(
                function<float(int, int)> samples,
                int numSamples) {
            cv::Mat sampleMat(numSamples, DIM, CV_64F);

            for (int sI = 0; sI < numSamples; sI++) {
                for (int d =0; d < DIM; d++) {
                    sampleMat.at<double>(sI, d) = (double) samples(sI, d);
                }
            }

            em.train(sampleMat);
        }

        inline float logLikelihood(
                function<float(int)> sample) {
            cv::Mat v(1, DIM, CV_64F);

            for (int d = 0; d < DIM; d++) {
                v.at<double>(0, d) = (double) sample(d);
            }

            return em.predict(v)[0];
        }

    private:
        cv::EM em;
};

