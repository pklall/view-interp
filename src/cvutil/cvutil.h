#pragma once

#include "common.h"

#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv/ml.h>
#include <opencv2/core/eigen.hpp>

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

inline void decomposeProjectionMatrix(
        const Eigen::Matrix<double, 3, 4>& P,
        Eigen::Matrix3d R,
        Eigen::Vector3d T) {
    cv::Mat cvP(P.rows(), P.cols(), CV_64FC1, (void*) P.data());
    cv::Mat cvK, cvR, cvT;

    cv::decomposeProjectionMatrix(cvP, cvK, cvR, cvT);

    cv::cv2eigen(cvR, R);
    cv::cv2eigen(cvT, T);
}

/*
 * Don't use this!  It's buggy!
inline void triangulate(
        const Eigen::Matrix<double, 3, 4>& cam0,
        const Eigen::Matrix<double, 3, 4>& cam1,
        const Eigen::Vector2d& match0,
        const Eigen::Vector2d& match1,
        Eigen::Vector4d& triangulated) {
    cv::Mat cam0CV(cam0.rows(), cam0.cols(), CV_64FC1, (void*) cam0.data());
    cv::Mat cam1CV(cam1.rows(), cam1.cols(), CV_64FC1, (void*) cam1.data());

    cv::Mat match0CV(1, 1, CV_64FC2, (void*) match0.data());
    cv::Mat match1CV(1, 1, CV_64FC2, (void*) match1.data());

    cv::Mat triCV(1, 1, CV_64FC4);

    cv::triangulatePoints(cam0CV, cam1CV, match0CV, match1CV, triCV);

    triangulated[0] = triCV.at<double>(1);
    triangulated[1] = triCV.at<double>(2);
    triangulated[2] = triCV.at<double>(3);
    triangulated[3] = triCV.at<double>(4);
}
*/


class CVOpticalFlow {
    public:
        CVOpticalFlow(
                int _wndSize,
                int _pyrLevels);

        void init(
                const CImg<uint8_t>& base,
                int maxFeatures,
                double minDistance);

        void compute(
                const CImg<uint8_t>& other);

        inline int featureCount() const {
            return goodFeatures.size();
        }

        inline bool getMatch(
                int i,
                Eigen::Vector2f& base,
                Eigen::Vector2f& match,
                float& error) const {
            base = Eigen::Vector2f(
                    goodFeatures[i].x,
                    goodFeatures[i].y);

            match = Eigen::Vector2f(
                    matches[i].x,
                    matches[i].y);

            error = matchError[i];

            return matchMask[i];
        }

    private:
        int wndSize;
        int pyrLevels;
        vector<cv::Mat> basePyr;
        vector<cv::Point2f> goodFeatures;
        vector<cv::Point2f> matches;
        vector<uint8_t> matchMask;
        vector<float> matchError;
};


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
                int _maxPoints,
                int _patchSize);

        int detectFeatures(
                const CImg<uint8_t>& grayImg);

        int match(
                const CVFeatureMatcher& other,
                vector<tuple<int, int>>& matchList,
                int maxMatches);

        int match(
                const CVFeatureMatcher& other,
                vector<tuple<float, float, float, float>>& matchedPoints);

        int match(
                const CVFeatureMatcher& other,
                bool sortByMatchScore,
                array<vector<cv::Point2f>, 2>& matchedPoints);

};

class CVFundamentalMatrixEstimator {
    public:
        void init(
                CVFeatureMatcher& left,
                CVFeatureMatcher& right);

        void init(
                const CVOpticalFlow& correspondence,
                double imgCenterX,
                double imgCenterY,
                double imgSize);

        inline void init() {
            points[0].clear();
            points[1].clear();
        }

        inline void addMatch(
                const Eigen::Vector2d& left,
                const Eigen::Vector2d& right) {
            points[0].push_back(cv::Point2f(left.x(), left.y()));
            points[1].push_back(cv::Point2f(right.x(), right.y()));
        }

        void estimateFundamentalMatrix(
                Eigen::Matrix3d& fundMat);

        void estimateEssentialMatrix(
                Eigen::Matrix3d& essentialMat,
                Eigen::Matrix3d& rotation,
                Eigen::Vector3d& translation);

        inline int getMatchCount() {
            return inlierMask.size();
        }

        inline bool isInlier(
                size_t index) {
            return inlierMask[index];
        }

        inline bool getMatch(
                size_t index,
                Eigen::Vector2d& left,
                Eigen::Vector2d& right) {
            if (inlierMask[index] == 0) {
                return false;
            }

            const auto& lp = points[0][index];
            const auto& rp = points[1][index];

            left = Eigen::Vector2d(lp.x, lp.y);
            right = Eigen::Vector2d(rp.x, rp.y);

            return true;
        }

    private:
        array<vector<cv::Point2f>, 2> points;
        vector<uint8_t> inlierMask;
};

class CVStereo {
    public:
        /**
         * Computes stereo on a rectified pair of the given dimensions with
         * pixels provided as grayscale in scanline order.
         * The resulting disparity, multiplied by a factor of 16 is stored
         * in resultBuf in scanline order.
         */
        static void stereo(
                int minDisparity,
                int numDisparities,
                int width,
                int height,
                const uint8_t* leftGray,
                const uint8_t* rightGray,
                int16_t* resultBuf);

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

