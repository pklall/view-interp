#include "common.h"

#include "cvutil.h"

#include <cstring>

void CVStereo::stereo(
        int minDisparity,
        int numDisparities,
        int width,
        int height,
        const uint8_t* leftGray,
        const uint8_t* rightGray,
        int16_t* resultBuf) {
    int rows = height;
    int cols = width;

    const cv::Mat left(rows, cols, CV_8U, (void*) leftGray);
    const cv::Mat right(rows, cols, CV_8U, (void*) rightGray);
    cv::Mat result(rows, cols, CV_16S, (void*) resultBuf);

    int SADWindowSize = 5; // 3 to 11 is recommended
    float smoothnessScale = 1.0f;
    int P1=8 * 3 * sqr(SADWindowSize) * smoothnessScale;
    int P2=32 * 3 * sqr(SADWindowSize) * smoothnessScale;
    int disp12MaxDiff=2;
    int preFilterCap=0;
    int uniquenessRatio=0;
    int speckleWindowSize=0;
    int speckleRange=0;
    bool fullDP=false;

    cv::StereoSGBM sgbm(minDisparity, numDisparities, SADWindowSize,
            P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize,
            speckleRange, fullDP);

    sgbm(left, right, result);
}

void CVStereo::rectify() {
    // Find interest points using ORB...

    std::vector<cv::KeyPoint> keypoints[2];
    cv::Mat descriptors[2];


    // Parameters for ORB features
    int nfeatures=500;
    float scaleFactor=1.2f;
    int nlevels=8;
    int edgeThreshold=31;
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=cv::ORB::HARRIS_SCORE;
    int patchSize=31;

    for (int i = 0; i < 2; i++) {
        cv::Mat grayCV;
        cv::Mat grayCV8;

        cv::ORB orb(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                WTA_K, scoreType, patchSize);

        cv::cvtColor(original[i], grayCV, CV_BGR2GRAY);

        grayCV.convertTo(grayCV8, CV_8U);

        orb(grayCV8, cv::Mat(), keypoints[i], descriptors[i]);
    }

    // Match interest points...


    int normType = cv::NORM_HAMMING;

    if (WTA_K == 3 || WTA_K == 4) {
        normType = cv::NORM_HAMMING2;
    }

    std::vector<cv::DMatch> matchList;

    cv::BFMatcher matcher(normType, true);

    matcher.match(descriptors[0], descriptors[1], matchList);

    // Compute the fundatmental matrix...

    // Convert the matched points into an appropriate format
    int numPoints = matchList.size();

    std::vector<cv::Point2f> points[2] {
        std::vector<cv::Point2f>(numPoints),
        std::vector<cv::Point2f>(numPoints)
    };

    for (int pt = 0; pt < numPoints; pt++) {
        points[0][pt] = cv::Point2f(
                keypoints[0][matchList[pt].queryIdx].pt.x,
                keypoints[0][matchList[pt].queryIdx].pt.y);
    }

    for (int pt = 0; pt < numPoints; pt++) {
        points[1][pt] = cv::Point2f(
                keypoints[1][matchList[pt].trainIdx].pt.x,
                keypoints[1][matchList[pt].trainIdx].pt.y);
    }

    warp(points);
}

void CVStereo::warp(std::vector<cv::Point2f> points[2]) {
    float inlierEpipolarMaxDist = 3;
    float targetConfidence = 0.99;

    cv::Mat fundMat = cv::findFundamentalMat(points[0], points[1],
            cv::FM_RANSAC, inlierEpipolarMaxDist, targetConfidence);

    cv::Size size(
            max(original[0].size().height,  original[1].size().height),
            max(original[0].size().width, original[1].size().width));

    cv::stereoRectifyUncalibrated(points[0], points[1], fundMat,
            size, rectTransforms[0], rectTransforms[1]);

    for (int i = 0; i < 2; i++) {
        rectified[i] = cv::Mat(original[i].size(), original[i].type());
        cv::warpPerspective(original[i], rectified[i],
                rectTransforms[i], original[i].size());
    }
}

void CVStereo::processPrerectified() {
    for (int i = 0; i < 2; i++) {
        rectTransforms[i] = cv::Mat::eye(3, 3, CV_32F);
        rectified[i] = original[i];
    }
}

CVStereo::CVStereo(
        const CImg<float>& left,
        const CImg<float>& right,
        bool prerectivied) {
    convertCImgToMat(left, original[0]);
    convertCImgToMat(right, original[1]);

    if (prerectivied) {
        processPrerectified();
    } else {
        rectify();
    }
}


void CVStereo::getRectified(
        CImg<float>& left,
        CImg<float>& right) {
    convertMatToCImg(rectified[0], left);
    convertMatToCImg(rectified[1], right);
}

void CVStereo::matchStereo(
        int minDisparity,
        int maxDisparity,
        int windowSize,
        float smoothnessScale) {
    this->minDisparity = minDisparity;
    this->numDisparities = maxDisparity - minDisparity;
    int SADWindowSize = windowSize; // 3 to 11 is recommended
    int P1=8 * 3 * sqr(SADWindowSize) * smoothnessScale;
    int P2=32 * 3 * sqr(SADWindowSize) * smoothnessScale;
    int disp12MaxDiff=2;
    int preFilterCap=0;
    int uniquenessRatio=0;
    int speckleWindowSize=0;
    int speckleRange=0;
    bool fullDP=false;

    cv::StereoSGBM sgbm(minDisparity, numDisparities, SADWindowSize,
            P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize,
            speckleRange, fullDP);

    cv::Mat left, right;

    rectified[0].convertTo(left, CV_8UC3);
    rectified[1].convertTo(right, CV_8UC3);

    sgbm(left, right, stereoDisparity);
}

void CVStereo::getStereo(CImg<float>& out) {
    int width = stereoDisparity.size().width;
    int height = stereoDisparity.size().height;

    CImg<int16_t> raw(width, height, 1, 1);

    memcpy(raw.data(), stereoDisparity.data, width * height * sizeof(int16_t));

    out = raw;

    out /= 16.0f;
}
