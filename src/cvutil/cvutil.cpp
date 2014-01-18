#include "cvutil.h"

#include "common.h"

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
        CImg<int>& result) {
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

    result = CImg<int>(labIn.width(), labIn.height());

    cimg_forXY(result, x, y) {
        result(x, y) = clusters[x][y];
    }
}

