#include "sparse_daisy_stereo.h"

#include <math.h>

void SparseDaisyStereo::init(
        const CImg<uint8_t>& mainImgGray) {
    if (!mainDaisy) {
        mainDaisy.reset(new daisy());
    }

    initializeDaisyForImg(mainImgGray, *mainDaisy);
}

void SparseDaisyStereo::match(
        const CImg<uint8_t>& otherImgGray,
        const PolarFundamentalMatrix& F,
        const vector<Eigen::Vector2f>& samples,
        vector<Eigen::Vector2f>& matches,
        vector<float>& matchDistances) {
    const Eigen::Vector2d& e0 = F.getEpipole(0);
    const Eigen::Vector2d& e1 = F.getEpipole(1);

    std::fill(matchDistances.begin(), matchDistances.end(),
            std::numeric_limits<float>::max());

    // Initialize a matrix with rows containing descriptors for each sample
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        DescriptorMat;

    typedef nanoflann::KDTreeEigenMatrixAdaptor<DescriptorMat> DescriptorMatKdTree;
    
    DescriptorMat descriptors(samples.size(), mainDaisy->descriptor_size());

    for (int i = 0; i < samples.size(); i++) {
        const float& x = samples[i].x();
        const float& y = samples[i].y();

        Eigen::Vector2d epipolarLine = samples[i].cast<double>() - e0;

        float orientation = (atan2(epipolarLine.y(), epipolarLine.x()) + PI) *
            180.0 / PI;
        
        mainDaisy->get_descriptor(y, x, orientation, &(descriptors(i, 0)));
    }

    DescriptorMatKdTree kdTree(mainDaisy->descriptor_size(), descriptors, 10);
    
    kdTree.index->buildIndex();

    // Query...

    daisy otherDaisy;

    initializeDaisyForImg(otherImgGray, otherDaisy);

    Eigen::VectorXf query(mainDaisy->descriptor_size());

    cimg_forXY(otherImgGray, x, y) {
        Eigen::Vector2d p(x, y);

        Eigen::Vector2d epipolarLine = p - e0;

        float orientation = (atan2(epipolarLine.y(), epipolarLine.x()) + PI) *
            180.0 / PI;
        
        mainDaisy->get_descriptor(y, x, orientation, query.data());

        size_t closest;
        float distance;

        kdTree.index->knnSearch(query.data(), 1, &closest, &distance);

        if (distance < matchDistances[closest]) {
            matches[closest].x() = x;
            matches[closest].y() = y;

            matchDistances[closest] = distance;
        }
    }
}

void SparseDaisyStereo::initializeDaisyForImg(
        const CImg<uint8_t>& img,
        daisy& d) {
    d.set_image(img.data(), img.height(), img.width());

    d.verbose(0);

    int rad = 15;
    int radq = 3;
    int thq = 8;
    int histq = 8;

    d.set_parameters(rad, radq, thq, histq);

    d.initialize_single_descriptor_mode();
}
