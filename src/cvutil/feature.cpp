#include "cvutil.h"

CVFeatureMatcher::CVFeatureMatcher(
        ID _id,
        int _maxPoints) :
    id(_id),
    keypoints(vector<cv::KeyPoint>(_maxPoints)) {
    descriptors = cv::Mat();

    // Parameters for ORB features
    int nfeatures=500;
    float scaleFactor=1.2f;
    int nlevels=8;
    int edgeThreshold=31;
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=cv::ORB::HARRIS_SCORE;
    int patchSize=31;

    normType = cv::NORM_HAMMING;

    if (WTA_K == 3 || WTA_K == 4) {
        normType = cv::NORM_HAMMING2;
    }

    orb = unique_ptr<cv::ORB>(new cv::ORB(nfeatures, scaleFactor, nlevels,
                edgeThreshold, firstLevel, WTA_K, scoreType, patchSize));
}

void CVFeatureMatcher::detectFeatures(
        const CImg<uint8_t>& grayImg) {
    keypoints.clear();

    cv::Mat img = cv::Mat(grayImg.height(), grayImg.width(), CV_8UC1,
            (void*) grayImg.data());

    (*orb)(img, cv::Mat(), keypoints, descriptors);

}

void CVFeatureMatcher::match(
        const CVFeatureMatcher& other,
        vector<tuple<ID, int, ID, int>>& matchList) {
    cvMatchList.clear();

    cv::BFMatcher matcher(normType, true);

    matcher.match(descriptors, other.descriptors, cvMatchList);

    for (const cv::DMatch& match : cvMatchList) {
        matchList.push_back(make_tuple(
                    id, match.trainIdx,
                    other.id, match.queryIdx));
    }
}
