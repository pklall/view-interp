#include <vector>
#include <string>
#include <functional>
#include <limits>
#include "stdio.h"
#include <cassert>

#include <omp.h>

#include "common.h"

#include "files.h"

using namespace std;

using namespace cimg_library;

extern template struct cimg_library::CImg<float>;

void featureMatchCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& matches);

void visualizeCorrespondence(
        const CImg<float>& from,
        const CImg<int>& corr,
        const CImg<float>& to) {
    CImgDisplay fromDisp(from);
    CImgDisplay corrDisp(corr.get_normalize(0, 255).get_equalize(0, 255));
    CImgDisplay toDisp(to);

    CImg<float> toAnnotated = to;

    while (!fromDisp.is_closed()) {
        fromDisp.wait();

        // mouse coordinates
        int mx = fromDisp.mouse_x();
        int my = fromDisp.mouse_y();

        toAnnotated = to;
        if (mx >= 0 && mx < corr.width() &&
                my >= 0 && my < corr.height()) {
            int corrX = mx + corr(mx, my, 0);
            int corrY = my + corr(mx, my, 1);

            float circleColor[3] = {255.0f, 0.0f, 0.0f};
            toAnnotated.draw_circle(corrX, corrY, 5, circleColor, 1.0f, 0);
        }

        toDisp.display(toAnnotated);
    }
}

void visualizeVectorImg(const CImg<float>& vec) {
    CImg<float> proj(vec.get_channel(0));

    CImgDisplay projDisp(proj);

    while (!projDisp.is_closed()) {
        projDisp.wait();
        // mouse coordinates
        int mx = projDisp.mouse_x();
        int my = projDisp.mouse_y();

        // mouse vector relative to center
        float mvx = mx - (projDisp.width()/2.0f);
        float mvy = my - (projDisp.height()/2.0f);
        float mvnorm = norm(mvx, mvy);

        // normalized mouse...
        float nmvx = mvx / mvnorm;
        float nmvy = mvy / mvnorm;

        // Update proj with the magnitude of the gradient onto
        // the mouse vector
        cimg_forXYZC(proj, x, y, z, c) {
            float xv = vec(x, y, 0);
            float yv = vec(x, y, 1);
            float px = xv * nmvx + yv * nmvy;
            proj(x, y, z, c) = px;
        }

        proj.display(projDisp);
    }
}

void displayClickable(CImg<float>& img,
        std::function<void(int, int)> onClick) {
    CImgDisplay disp(img);
    while (!disp.is_closed()) {
        disp.wait();
        if (!(disp.button() && disp.mouse_y()>=0)) {
            continue;
        }

        onClick(disp.mouse_x(), disp.mouse_y());

        disp.display(img);
    }
}

/**
 * Implementation of extremely-generalized PatchMatch optimization of
 * arbitrary-dimension parameters over a 2D grid.
 *
 * Note that `field`, and `dist` must have the same width and height.
 *
 * `dist` must be 1 dimensional.
 *
 * `field` may have spectrum of arbitrary size, each channel representing
 * one element of the vector to optimize per pixel.  For example, a 2-channel
 * field may be used to optimize translational displacements.
 *
 * randomSample(int x, int y, int iteration, float[] value) must randomly mutate the given
 * value.  Note that the iteration number is included to allow for progressive
 * refinement.
 *
 * patchDist(int x, int y, float[] value) must return the error resulting
 * from assigning `value` to `field(x, y)`
 *
 * Returns the number of modified values.
 */
inline int patchMatch(
        CImg<float>& field,
        CImg<float>& dist,
        function<void(int, int, int, float[])> randomSample,
        function<float(int, int, float[])> patchDist,
        int iterations) {
    int propDirection = 1;

    int totalNumChanged = 0;

    for (int iter = 0; iter < iterations; iter++) {
        int numChanged = 0;
        // Switch propagation direction during each iteration.
        propDirection *= -1;

#pragma omp parallel for shared(numChanged)
        for (int y = 0; y < field.height(); y++) {
            for (int x = 0; x < field.width(); x++) {
                // propagation
                // try the adjacent pixels along propDirection
                int adjY = y + propDirection;
                int adjX = x + propDirection;

                float tmp[field.spectrum()];

                if (adjY >= 0 && adjY < field.height()) {
                    bool different = false;

                    cimg_forC(field, c) {
                        different |= field(x, adjY, c) != field(x, y, c);
                        tmp[c] = field(x, adjY, c);
                    }

                    if (different) {
                        float d = patchDist(x, y, tmp);

                        if (d < dist(x, y)) {
                            dist(x, y) = d;
                            cimg_forC(field, c) {
                                field(x, y, c) = tmp[c];
                            }
#pragma omp critical
                            {
                                numChanged++;
                            }
                        }
                    }
                }

                if (adjX >= 0 && adjX < field.width()) {
                    bool different = false;

                    cimg_forC(field, c) {
                        different |= field(adjX, y, c) != field(x, y, c);
                        tmp[c] = field(adjX, y, c);
                    }

                    if (different) {
                        float d = patchDist(x, y, tmp);

                        if (d < dist(x, y)) {
                            dist(x, y) = d;
                            cimg_forC(field, c) {
                                field(x, y, c) = tmp[c];
                            }
#pragma omp critical
                            {
                                numChanged++;
                            }
                        }
                    }
                }

                
                // Random search
                cimg_forC(field, c) {
                    tmp[c] = field(x, y, c);
                }

                randomSample(x, y, iter, tmp);

                float d = patchDist(x, y, tmp);

                if (d < dist(x, y)) {
                    dist(x, y) = d;
                    cimg_forC(field, c) {
                        field(x, y, c) = tmp[c];
                    }
#pragma omp critical
                    {
                        numChanged++;
                    }
                }
            }
        }

        printf("PatchMatch iteration %d/%d, numChanged = %d\n", iter, iterations, numChanged);

        totalNumChanged += numChanged;
    }
    return totalNumChanged;
}

int patchMatchTranslationCorrespondence(
        const CImg<float>& img1,
        const CImg<float>& img2,
        CImg<float>& field,
        CImg<float>& dist,
        int wndSize = 15,
        int iterations = 5,
        float randomSearchFactor = 1.0f,
        bool recomputeDist = false) {
    CImg<float> lab1 = img1.get_RGBtoLab();
    CImg<float> lab2 = img2.get_RGBtoLab();

    auto sample = [&lab1, &lab2, wndSize, randomSearchFactor]
        (int sx, int sy, int iter, float* value) {
            float searchWndRadiusFactor = randomSearchFactor / (iter + 1);

            float searchWndWidth  = searchWndRadiusFactor * lab2.width();
            float searchWndHeight = searchWndRadiusFactor * lab2.height();

            float minSearchWndX = (sx - searchWndWidth / 2.0f);
            float minSearchWndY = (sy - searchWndHeight / 2.0f);

            float maxSearchWndX = (sx + searchWndWidth / 2.0f);
            float maxSearchWndY = (sy + searchWndHeight / 2.0f);

            minSearchWndX = max(0.0f, minSearchWndX);
            minSearchWndY = max(0.0f, minSearchWndY);

            maxSearchWndX = max((float) lab2.width(), maxSearchWndX);
            maxSearchWndY = max((float) lab2.height(), maxSearchWndY);

            // The point we have chosen to randomly sample from
            int randX = (int) (cimg::rand() * (maxSearchWndX - minSearchWndX) + minSearchWndX);
            int randY = (int) (cimg::rand() * (maxSearchWndY - minSearchWndY) + minSearchWndY);

            value[0] = randX - sx;
            value[1] = randY - sy;
        };

    auto patchDist = [&lab1, &lab2, wndSize]
        (int sx, int sy, float* value) -> float {
            int dx = sx + (int) value[0];
            int dy = sy + (int) value[1];

            // if (dx < 0 || dx >= lab2.width() ||
                    // dy < 0 || dy >= lab2.height()) {
                // return numeric_limits<float>::infinity();
            // }
            
            if (dx - wndSize / 2 < 0 || dx + wndSize / 2 >= lab2.width() ||
                    dy -wndSize / 2 < 0 || dy + wndSize / 2 >= lab2.height()) {
                return numeric_limits<float>::infinity();
            }

            int minSX = max(0, sx - wndSize / 2);
            int maxSX = min(lab1.width() - 1, sx + wndSize / 2);

            int minSY = max(0, sy - wndSize / 2);
            int maxSY = min(lab1.height() - 1, sy + wndSize / 2);

            int minDX = max(0, dx - wndSize / 2);
            int maxDX = min(lab2.width() - 1, dx + wndSize / 2);

            int minDY = max(0, dy - wndSize / 2);
            int maxDY = min(lab2.height() - 1, dy + wndSize / 2);

            // The extent of the valid window around (sx, sy) and (dx, dy)
            // to compare.
            int minX = -min(sx - minSX, dx - minDX);
            int maxX = min(maxSX - sx, maxDX - dx);

            int minY = -min(sy - minSY, dy - minDY);
            int maxY = min(maxSY - sy, maxDY - dy);

            float totalWeight = 0.0f;
            float ssd = 0.0f;
            for (int y = minY; y <= maxY; y++) {
                for (int x = minX; x <= maxX; x++) {
                    // Weight pixels with a bilateral-esque filter
                    float lab1Diff = 0.0f;
                    cimg_forZC(lab1, z, c) {
                        float lDiff = lab1(x + sx, y + sy, z, c) -
                            lab1(sx, sy, z, c);
                        lab1Diff = lDiff * lDiff;
                    }
                    float weight = exp(-(lab1Diff) / 30.0f);

                    cimg_forZC(lab1, z, c) {
                        float diff =
                            lab1(x + sx, y + sy, z, c) -
                            lab2(x + dx, y + dy, z, c);

                        totalWeight += weight;

                        ssd += diff * diff * weight;
                    }
                }
            }

            return ssd / totalWeight;
        };

    if (recomputeDist) {
#pragma omp parallel for
        for (int y = 0; y < dist.height(); y++) {
            for (int x = 0; x < dist.width(); x++) {
                float tmp[field.spectrum()];

                cimg_forC(field, c) {
                    tmp[c] = field(x, y, c);
                }

                float d = patchDist(x, y, tmp);
                dist(x, y) = d;
            }
        }
    }

    return patchMatch(field, dist, sample, patchDist, iterations);
}

/**
 * Same as patchMatchTranslationalCorrespondence, but solves for
 * an affine transform where field contains 6 values (a, b, c, d, e, f)
 * and the effective translation for each point (srcX, srcY) -> (dstX, dstY) is
 * (dstX, dstY) = (srcX + a + b * srcX + c * srcY),
 *                 srcY + d + e * srcX + f * srcY)
 *
 * Note that this is NOT the same as PatchMatch Stereo since it doens't assume
 * that img1 and img2 have been rectified.
 */
void patchMatchAffine(
        const CImg<float>& img1,
        const CImg<float>& img2,
        CImg<float>& field,
        CImg<float>& dist,
        int wndSize = 15,
        int iterations = 5,
        float randomSearchFactor = 1.0f,
        bool recomputeDist = false) {
    CImg<float> lab1 = img1.get_RGBtoLab();
    CImg<float> lab2 = img2.get_RGBtoLab();

    auto sample = [&lab1, &lab2, wndSize, randomSearchFactor]
        (int sx, int sy, int iter, float* value) {
            float b, c, e, f;

            b = cimg::rand() * 1.0f - 0.5f;
            c = cimg::rand() * 1.0f - 0.5f;
            e = cimg::rand() * 1.0f - 0.5f;
            f = cimg::rand() * 1.0f - 0.5f;

            b = c = e = f = 0;

            float aCenter = sx + (b * sx + c * sy);
            float dCenter = sy + (e * sx + f * sy);

            float searchWndRadiusFactor = randomSearchFactor / (iter + 1);

            float searchWndWidth  = searchWndRadiusFactor * lab2.width();
            float searchWndHeight = searchWndRadiusFactor * lab2.height();

            float minSearchWndX = (aCenter - searchWndWidth / 2.0f);
            float minSearchWndY = (dCenter - searchWndHeight / 2.0f);

            float maxSearchWndX = (aCenter + searchWndWidth / 2.0f);
            float maxSearchWndY = (dCenter + searchWndHeight / 2.0f);

            minSearchWndX = max(0.0f, minSearchWndX);
            minSearchWndY = max(0.0f, minSearchWndY);

            maxSearchWndX = max((float) lab2.width(), maxSearchWndX);
            maxSearchWndY = max((float) lab2.height(), maxSearchWndY);

            int randX = (int) (cimg::rand() * (maxSearchWndX - minSearchWndX) + minSearchWndX);
            int randY = (int) (cimg::rand() * (maxSearchWndY - minSearchWndY) + minSearchWndY);

            value[0] = randX;
            value[1] = b;
            value[2] = c;
            value[3] = randY;
            value[4] = e;
            value[5] = f;
        };

    auto patchDist = [&lab1, &lab2, wndSize]
        (int sx, int sy, float* value) -> float {
            float ssd = 0.0f;
            float totalWeight = 0.0f;

            for (int y = wndSize / 2; y <= wndSize / 2; y++) {
                for (int x = wndSize / 2; x <= wndSize / 2; x++) {
                    int srcX = x + sx;
                    int srcY = y + sy;

                    float dstX = srcX +
                        value[0] + value[1] * srcX + value[2] * srcY;
                    float dstY = srcY +
                        value[3] + value[4] * srcX + value[5] * srcY;
                    
                    float lab1Diff = 0.0f;
                    cimg_forZC(lab1, z, c) {
                        float lDiff = lab1(srcX, srcY, z, c) -
                            lab1(sx, sy, z, c);
                        lab1Diff += lDiff * lDiff;
                    }
                    float weight = exp(-(lab1Diff) / 30.0f);

                    if (srcX >= 0 && srcX < lab1.width() &&
                            srcY >= 0 && srcY < lab1.height() &&
                            dstX >= 0 && dstX < lab2.width() &&
                            dstY >= 0 && dstY < lab2.height()) {
                        cimg_forZC(lab1, z, c) {
                            float diff = lab1(srcX, srcY, z, c) -
                                lab2.linear_atXYZC(dstX, dstY, z, c);
                            // lab1(x + sx, y + sy, z, c) -
                            // lab2(x + dx, y + dy, z, c);
                            ssd += diff * diff * weight;
                            totalWeight += weight;
                        }
                    }
                }
            }
            if (totalWeight == 0.0f) {
                return std::numeric_limits<float>::max();
            }

            return ssd / totalWeight;
        };

    if (recomputeDist) {
#pragma omp parallel for
        for (int y = 0; y < dist.height(); y++) {
            for (int x = 0; x < dist.width(); x++) {
                float tmp[field.spectrum()];

                cimg_forC(field, c) {
                    tmp[c] = field(x, y, c);
                }

                float d = patchDist(x, y, tmp);
                dist(x, y) = d;
            }
        }
    }

    patchMatch(field, dist, sample, patchDist, iterations);
}

inline void pyramidPatchMatch(
        const CImg<float>& img1,
        const CImg<float>& img2,
        CImg<float>& field,
        CImg<float>& dist,
        int wndSize = 30,
        int iterations = 5,
        int levels = 3) {
    if (levels > 1) {
        CImg<float> fieldHalf = field.get_resize_halfXY();
        CImg<float> distHalf = dist.get_resize_halfXY();
        CImg<float> img1Half = img1.get_resize_halfXY();
        CImg<float> img2Half = img2.get_resize_halfXY();

        pyramidPatchMatch(img1Half, img2Half, fieldHalf, distHalf,
                wndSize, iterations, levels - 1);

        field = fieldHalf.get_resize_doubleXY();
    }

    printf("Processing pyramid level: %d\n", levels);
    // patchMatchAffine(img1, img2, field, dist,
            // wndSize, iterations, 1.0f / levels, true);
    patchMatchTranslationCorrespondence(img1, img2, field, dist,
            wndSize, iterations, 1.0f / levels, true);
}

int main(int argc, char** argv) {
    for (int sampleIndex = 0; sampleIndex < NUM_SAMPLES; sampleIndex++) {
        CImg<float> fst(SAMPLES[sampleIndex][0].c_str());
        CImg<float> lst(SAMPLES[sampleIndex][1].c_str());

        while (fst.width() > 1024 || fst.height() > 1024) {
            fst.resize_halfXY();
            lst.resize_halfXY();
        }

        CImg<float> corr(fst.width(), fst.height(), 1, 2);
        CImg<float> error(fst.width(), fst.height());

        corr = 0.0f;
        error = std::numeric_limits<float>::infinity();

        // patchMatchFieldDist(fst, lst, corr, error, 7, 9);
        // patchMatchTranslationCorrespondence(fst, lst, corr, error, 15, 5, 0.5f);
        
        // pyramidPatchMatch(fst, lst, corr, error, 21, 5, 2);

        // visualizeCorrespondence(fst, corr, lst);
        CImg<float> matches;
        featureMatchCV(
                fst.get_RGBtoLab().channel(0),
                lst.get_RGBtoLab().channel(0),
                matches);

        matches.display();

        /*
        CImgDisplay fstDisp(fst);
        // CImgDisplay lstDisp(lst);
        // CImgDisplay errDisp(error);
        CImgList<float>(
                corr.get_channel(0).get_equalize(255),
                corr.get_channel(1).get_equalize(255)
                ).display();
        */
    }

    return 1;
}


/**
 * Computes optical flow from a to b and places the displacement field
 * into flow with x and y components in depth planes 0 and 1, respectively.
 */
void flowLK(
        const CImg<float>& a,
        const CImg<float>& b,
        float sigma,
        CImg<float>& flow) {
    assert(a.width() == b.width() && a.width() == flow.width() &&
            a.height() == b.height() && a.height() == flow.height());

    assert(flow.depth() == 2);

    // This uses the least-squares method shown in
    // http://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method

    CImgList<float> spaceGrad = a.get_gradient();
    const CImg<float>& Ix = spaceGrad(0);
    const CImg<float>& Iy = spaceGrad(1);

    CImg<float> It(a.width(), a.height());
    It = b - a;

    CImg<float> A00 = Ix.get_sqr();
    CImg<float> Adiag = Ix.get_mul(Iy);
    CImg<float> A11 = Iy.get_sqr();

    CImg<float> B0 = Ix.get_mul(It) * -1;
    CImg<float> B1 = Iy.get_mul(It) * -1;

    // Apply weighting function, this is equivalent to blurring for which
    // CImg has a faster-than-naive-gaussian implementation.
    A00.blur(sigma);
    Adiag.blur(sigma);
    A11.blur(sigma);

    // Compute results by inverting A and multiplying with B
    cimg_forXY(flow, x, y) {
        float det = A00(x, y) * A11(x, y) - Adiag(x, y) * Adiag(x, y);
        float AInv00   =  A11(x, y)   / det;
        float AInvdiag = -Adiag(x, y) / det;
        float AInv11   =  A00(x, y)   / det;

        flow(x, y, 0) = AInv00   * B0(x, y) + AInvdiag * B1(x, y);
        flow(x, y, 1) = AInvdiag * B1(x, y) + AInv11   * B1(x, y);
    }
}

void stereoBMCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& flow) {
    assert(a.depth() == 1 && a.spectrum() == 1 &&
            b.depth() == 1 && b.spectrum() == 1);

    assert(a.width() == b.width() && a.width() == flow.width() &&
            a.height() == b.height() && a.height() == flow.height());

    assert(flow.depth() == 2);

    // Normalize before passing to OpenCV
    CImg<float> aNorm = a.get_normalize(0.0f, 1.0f);
    CImg<float> bNorm = b.get_normalize(0.0f, 1.0f);

    // cv::GpuMat cvA(aNorm.height(), aNorm.width(), CV_32FC1, (void*) aNorm.data());

    // cv::GpuMat cvB(bNorm.height(), bNorm.width(), CV_32FC1, (void*) bNorm.data());

    // cv::GpuMat cvFlow(flow.height(), flow.width(), CV_32FC2, (void*) flow.data);

    // cv::StereoConstantSpaceBP bp;

    // bp(cvA, cvB, cvFlow);

}

void flowCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& flow) {
    assert(a.depth() == 1 && a.spectrum() == 1 &&
            b.depth() == 1 && b.spectrum() == 1);

    assert(a.width() == b.width() && a.width() == flow.width() &&
            a.height() == b.height() && a.height() == flow.height());

    assert(flow.depth() == 2);

    // Normalize before passing to OpenCV
    CImg<float> aNorm = a.get_normalize(0.0f, 1.0f);
    CImg<float> bNorm = b.get_normalize(0.0f, 1.0f);

    cv::Mat cvA(aNorm.height(), aNorm.width(), CV_32FC1, (void*) aNorm.data());

    cv::Mat cvB(bNorm.height(), bNorm.width(), CV_32FC1, (void*) bNorm.data());

    cv::Mat cvFlow(flow.height(), flow.width(), CV_32FC2);

    double pyr_scale = 0.5;
    int levels = 10;
    int winsize = 3;
    int iterations = 5;
    int poly_n = 5;
    double poly_sigma = 1.5;
    int flags = 0;
    cv::calcOpticalFlowFarneback(cvA, cvB, cvFlow, 
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

    cv::Mat cvFlowX(flow.height(), flow.width(), CV_32FC1,
            (void*) flow.data());
    cv::Mat cvFlowY(flow.height(), flow.width(), CV_32FC1,
            (void*) (flow.data() + flow.width() * flow.height()));

    std::vector<cv::Mat> output;
    output.push_back(cvFlowX);
    output.push_back(cvFlowY);

    cv::split(cvFlow, output);
}

/**
 * Uses OpenCV to match interest points in Grayscale images 'a' and 'b'.
 *
 * The result is returned in matches as a set of column vectors containing
 * (a.x, a.y, b.x, b.y).
 *
 * See http://docs.opencv.org/doc/tutorials/features2d/feature_description/feature_description.html#feature-description
 */
void featureMatchCV(
        const CImg<float>& a,
        const CImg<float>& b,
        CImg<float>& matches) {
    assert(a.depth() == 1 && a.spectrum() == 1 &&
            b.depth() == 1 && b.spectrum() == 1);

    CImg<float> aNorm = a.get_normalize(0.0f, 255.0f);
    CImg<float> bNorm = b.get_normalize(0.0f, 255.0f);

    cv::Mat cvA(aNorm.height(), aNorm.width(), CV_32FC1, (void*) aNorm.data());

    cv::Mat cvB(bNorm.height(), bNorm.width(), CV_32FC1, (void*) bNorm.data());

    cv::Mat cvA8;
    cvA.convertTo(cvA8, CV_8U);

    cv::Mat cvB8;
    cvB.convertTo(cvB8, CV_8U);

    cv::ORB orb;

    std::vector<cv::KeyPoint> keypointsA, keypointsB;
    cv::Mat descriptorsA, descriptorsB;

    orb(cvA8, cv::Mat(), keypointsA, descriptorsA);
    orb(cvB8, cv::Mat(), keypointsB, descriptorsB);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matchList;
    matcher.match(descriptorsA, descriptorsB, matchList);

    matches = CImg<float>(matchList.size(), 4);

    cimg_forX(matches, i) {
        matches(i, 0) = keypointsA[matchList[i].queryIdx].pt.x;
        matches(i, 1) = keypointsA[matchList[i].queryIdx].pt.y;
        matches(i, 2) = keypointsB[matchList[i].trainIdx].pt.x;
        matches(i, 3) = keypointsB[matchList[i].trainIdx].pt.y;
    }
}
