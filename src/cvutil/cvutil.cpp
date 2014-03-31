#include "cvutil.h"

#include "common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

#include "slic.h"

void convertCImgToMat(
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

void convertMatToCImg(
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

void slicSuperpixels(
        const CImg<float>& labIn,
        int numSuperpixels,
        int nc,
        CImg<uint16_t>& result) {
    int w = labIn.width();
    int h = labIn.height();

    double step = sqrt((w * h) / (double) numSuperpixels);

    cv::Mat labInCV;

    convertCImgToMat(labIn, labInCV);

    Slic slic;
    IplImage labInCVIpl = labInCV;
    slic.generate_superpixels(&labInCVIpl, step, nc);
    slic.create_connectivity(&labInCVIpl);
    
    vector<vector<int>>& clusters = slic.getClusters();

    result = CImg<uint16_t>(labIn.width(), labIn.height());

    cimg_forXY(result, x, y) {
        result(x, y) = clusters[x][y];
    }
}

CVOpticalFlow::CVOpticalFlow(
        int _wndSize,
        int _pyrLevels) :
    wndSize(_wndSize),
    pyrLevels(_pyrLevels) {
}

void CVOpticalFlow::init(
        const CImg<uint8_t>& base,
        int maxFeatures,
        double minDistance) {
    cv::Mat baseCV = cv::Mat(base.height(), base.width(), CV_8UC1,
            (void*) base.data());

    cv::goodFeaturesToTrack(baseCV, goodFeatures, maxFeatures, 0.01, minDistance); 
    
    cv::cornerSubPix(baseCV, goodFeatures, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 5));

    cv::buildOpticalFlowPyramid(baseCV, basePyr, cv::Size(wndSize, wndSize),
            pyrLevels);

    matches.resize(goodFeatures.size());
    matchMask.resize(goodFeatures.size());
    matchError.resize(goodFeatures.size());
}

void CVOpticalFlow::compute(
        const CImg<uint8_t>& other) {
    cv::Mat otherCV = cv::Mat(other.height(), other.width(), CV_8UC1,
            (void*) other.data());

    cv::calcOpticalFlowPyrLK(basePyr, otherCV, goodFeatures, matches, matchMask,
            matchError);
}

void CVFundamentalMatrixEstimator::init(
        CVFeatureMatcher& left,
        CVFeatureMatcher& right) {
    points[0].clear();
    points[1].clear();

    inlierMask.clear();

    left.match(right, false, points);
}

void CVFundamentalMatrixEstimator::init(
        const CVOpticalFlow& correspondence,
        double imgCenterX,
        double imgCenterY,
        double imgSize) {
    points[0].clear();
    points[1].clear();

    inlierMask.clear();

    for (int pointI = 0; pointI < correspondence.featureCount(); pointI++) {
        Eigen::Vector2f pt0;
        Eigen::Vector2f pt1;
        float error;

        if (correspondence.getMatch(pointI, pt0, pt1, error)) {
            points[0].push_back(
                    cv::Point2f(
                        (pt0.x() - imgCenterX) / imgSize,
                        (pt0.y() - imgCenterY) / imgSize));
            points[1].push_back(
                    cv::Point2f(
                        (pt1.x() - imgCenterX) / imgSize,
                        (pt1.y() - imgCenterY) / imgSize));
        }
    }
}

void CVFundamentalMatrixEstimator::estimateFundamentalMatrix(
        Eigen::Matrix3d& fundMat) {
    float inlierEpipolarMaxDist = 1;
    float targetConfidence = 0.99999;

    inlierMask.resize(points[0].size());

    cv::Mat cvFundMat = cv::findFundamentalMat(points[0], points[1],
            cv::FM_LMEDS, inlierEpipolarMaxDist, targetConfidence,
            inlierMask);

    cv::cv2eigen(cvFundMat, fundMat);
}

