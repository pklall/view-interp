#include "common.h"

#include "cvstereo.h"

#include <cstring>

void CVStereo::convertCImgToMat(
        const CImg<float>& in,
        cv::Mat& out) {
    assert(in.spectrum() == 3);

    std::vector<cv::Mat> channels;

    // We must push in reverse order since CImg uses RGB, but OpenCV uses BGR
    for (int c = 2; c >= 0; c--) {
        auto cimgChannel = in.get_channel(c);

        cv::Mat m = cv::Mat(in.height(), in.width(), CV_32FC1, cimgChannel.data());

        // Note that we *must* clone since the constructor above doesn't copy
        // the data from cimgChannel, which will soon be deallocated.
        channels.push_back(m.clone());
    }


    cv::merge(channels, out);
}

void CVStereo::convertMatToCImg(
        const cv::Mat& in,
        CImg<float>& out) {
    cv::Mat inF;

    in.convertTo(inF, CV_32FC3);

    std::vector<cv::Mat> channels(3);

    cv::split(inF, channels);

    int width = in.size().width;
    int height = in.size().height;
    out = CImg<float>(width, height, 1, 3);

    for (int c = 0; c < 3; c++) {
        memcpy(out.get_shared_channel(2 - c).data(), channels[c].data,
                width * height * sizeof(float));
    }
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

void CVStereo::matchStereo() {
    minDisparity = -256;
    numDisparities = 512;
    int SADWindowSize = 3; // 3 to 11 is recommended
    int P1=8 * 3 * sqr(SADWindowSize);
    int P2=32 * 3 * sqr(SADWindowSize);
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
