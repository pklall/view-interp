#include <vector>
#include <string>
#include <functional>
#include <limits>
#include "stdio.h"
#include <cassert>

#include "common.h"

using namespace std;

using namespace cimg_library;

extern template struct cimg_library::CImg<float>;

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
 * Implementation of PatchMatch generalized to arbitrary
 * patch comparison functions.
 *
 * patchDist(srcX, srcY, dstX, dstY) must return the "distance" between
 * patches with the top-left corner at (dstX, dstY) and (dstX, dstY) in the
 * src and dst images, respectively.
 *
 * Returns nnf, the nearest-neighbor field.
 *
 * nnf(sx, sy) = (dx, dy, dist) indicating that the (dx, dy) in dest is
 * closest to (dx, dy) in src, differing by dist.  Note that nnf is not defined
 * for the boundary of (wndSize - 1) pixels on the bottom and right sides
 * of the image.
 *
 */
tuple<CImg<int>, CImg<float>> patchMatch(
        const CImg<float>& src,
        const CImg<float>& dst,
        function<float(int, int, int, int)> patchDist,
        int wndSize,
        int iterations) {
    // The nearest-neighbor field stores a 2-tuple (x delta, y delta)
    CImg<int> nnf(src.width() - wndSize + 1,
                  src.height() - wndSize + 1,
                  1,
                  2);

    CImg<float> bestDist(nnf.width(), nnf.height());

    // Initialize with a random field and associated distances
    cimg_forXY(nnf, x, y) {
        nnf(x, y, 0) = x;
        nnf(x, y, 1) = y;
        // nnf(x, y, 0) = (int) (cimg::rand() * (dst.width() - 1));
        // nnf(x, y, 1) = (int) (cimg::rand() * (dst.height() - 1));

        bestDist(x, y) =
            patchDist(x, y, nnf(x, y, 0), nnf(x, y, 1));
    }

    int propDirection = 1;
    for (int iter = 0; iter < iterations; iter++) {
        // Switch propagation direction during each iteration.
        propDirection *= -1;
        
        // propagation
        cimg_forXY(nnf, x, y) {
            // try the adjacent pixels along propDirection
            int adjY = y + propDirection;
            int adjX = x + propDirection;

            if (adjY >= 0 && adjY < nnf.height()) {
                int adjDx = nnf(x, adjY, 0);
                int adjDy = nnf(x, adjY, 1) - propDirection;

                if (adjDx >= 0 && adjDx < dst.width() &&
                    adjDy >= 0 && adjDy < dst.height()) {

                    float dist = patchDist(x, y, adjDx, adjDy);

                    if (dist < bestDist(x, y)) {
                        bestDist(x, y) = dist;
                        nnf(x, y, 0) = adjDx;
                        nnf(x, y, 1) = adjDy;
                    }
                }
            }

            if (adjX >= 0 && adjX < nnf.width()) {
                int adjDx = nnf(adjX, y, 0) - propDirection;
                int adjDy = nnf(adjX, y, 1);

                if (adjDx >= 0 && adjDx < dst.width() &&
                    adjDy >= 0 && adjDy < dst.height()) {

                    float dist = patchDist(x, y, adjDx, adjDy);

                    if (dist < bestDist(x, y)) {
                        bestDist(x, y) = dist;
                        nnf(x, y, 0) = adjDx;
                        nnf(x, y, 1) = adjDy;
                    }
                }
            }
        }
        
        // Random search
        cimg_forXY(nnf, x, y) {
            float searchWndRadiusFactor = 1.0f / pow(2.0f, iter);

            float searchWndWidth = searchWndRadiusFactor * src.width();
            float searchWndHeight= searchWndRadiusFactor * src.height();

            int minSearchWndX = (int) (nnf(x, y, 0) - searchWndWidth);
            int minSearchWndY = (int) (nnf(x, y, 1) - searchWndHeight);

            int maxSearchWndX = (int) (nnf(x, y, 0) + searchWndWidth);
            int maxSearchWndY = (int) (nnf(x, y, 1) + searchWndHeight);

            // Clamp the search box to the image.
            minSearchWndX = min(max(minSearchWndX, 0),
                                nnf.width() - 1);
            minSearchWndY = min(max(minSearchWndY, 0),
                                nnf.height() - 1);
            maxSearchWndX = min(max(maxSearchWndX, 0),
                                nnf.width() - 1);
            maxSearchWndY = min(max(maxSearchWndY, 0),
                                nnf.height() - 1);

            // The point we have chosen to randomly sample from
            int randX = (int) (minSearchWndX +
                    cimg::rand() * (maxSearchWndX - minSearchWndX));
            int randY = (int) (minSearchWndY +
                    cimg::rand() * (maxSearchWndY - minSearchWndY));

            float dist = patchDist(x, y, randX, randY);

            if (dist < bestDist(x, y)) {
                bestDist(x, y) = dist;
                nnf(x, y, 0) = randX;
                nnf(x, y, 1) = randY;
            }
        }
    }

    return tuple<CImg<int>, CImg<float>>(nnf, bestDist);
}

void patchMatchFieldDist(
        const CImg<float>& img1,
        const CImg<float>& img2,
        CImg<int>& field,
        CImg<float>& dist,
        int wndSize = 15,
        int iterations = 5) {
    const CImg<float> lab1 = img1.get_RGBtoLab();
    const CImg<float> lab2 = img2.get_RGBtoLab();

    // CImgList<float> grad1 = lab1.get_channel(0).get_gradient();
    // CImgList<float> grad2 = lab2.get_channel(0).get_gradient();

    // Weight each component differently
    float labWeight[3]{1.0f, 1.0f, 1.0f};
    auto patchDist = [&lab1, &lab2, wndSize]
    (int sx, int sy, int dx, int dy) -> float {
        float ssd = 0.0f;
        cimg_forZC(lab1, z, c) {
            for (int y = 0; y < wndSize; y++) {
                for (int x = 0; x < wndSize; x++) {
                    float diff = lab1(x + sx, y + sy, z, c) -
                                 lab2(x + dx, y + dy, z, c);
                    ssd += diff * diff * labWeight[c];
                }
            }
        }
        return ssd;
    };

    tuple<CImg<int>, CImg<float>> nnf_dist = patchMatch(
            img1, img2, patchDist, wndSize, iterations);

    // These are smaller than img1 because they only
    // contain valid points (the right and bottom border are empty)
    const CImg<int>& nnfKnown = get<0>(nnf_dist);
    const CImg<float>& distKnown = get<1>(nnf_dist);

    // For consistancy, the returned nnf and distance will be
    // of the same size as img1.
    // Also, the resulting field will only hold relative offsets

    field = CImg<int>(img1.width(), img1.height(), 2);
    field = 0;
    dist = CImg<float>(img1.width(), img1.height(), 1);
    dist = numeric_limits<float>::max();

    cimg_forXY(nnfKnown, x, y) {
        field(x, y, 0) = nnfKnown(x, y, 0) - x;
        field(x, y, 1) = nnfKnown(x, y, 1) - y;
        dist(x, y) = distKnown(x, y);
    }
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

int main(int argc, char** argv) {
    string files[] = {"samples/hall0.jpg",
                      "samples/hall1.jpg",
                      "samples/synth0.png",
                      "samples/synth1.png",
                      "samples/poster0.ppm",
                      "samples/poster1.ppm",
                      "samples/bed0.jpg",
                      "samples/bed1.jpg"};

    CImg<float> fst(files[6].c_str());
    CImg<float> lst(files[7].c_str());

    CImg<int> corr;
    CImg<float> error;

    while (fst.width() > 512 || fst.height() > 512) {
        fst.resize_halfXY();
        lst.resize_halfXY();
    }

    patchMatchFieldDist(fst, lst, corr, error, 7, 7);

    CImgList<float>(fst, lst, corr.get_equalize(255), error).display();

    /*
    // Flow based experiments:

    // Isolate luminance from color.
    fst.RGBtoLab();
    lst.RGBtoLab();

    while (fst.width() > 2048 || fst.height() > 2048) {
        fst.resize_halfXY();
        lst.resize_halfXY();
    }

    CImg<float> fstL = fst.get_channel(0);
    CImg<float> lstL = lst.get_channel(0);
    
    CImg<float> flow(fstL.width(), fstL.height(), 2);

    flowCV(fstL, lstL, flow);

    // flowLK(fstL, lstL, 8.0f, flow);

    CImg<float> flowMag = 
        (
         flow.get_shared_slice(0).get_sqr() +
         flow.get_shared_slice(1).get_sqr()
        ).sqrt();

    CImgList<float> grad = fstL.get_gradient();
    CImg<float> gradMag =
        (grad(0).get_sqr() + grad(1).get_sqr()).sqrt();

    CImg<float> flowMagMultGrad = flowMag.get_mul(gradMag);
    flowMagMultGrad.blur(3.0f);
    CImg<float> filteredFlowMag = flowMagMultGrad.div(gradMag.get_blur(3.0f));

    CImg<float> logMag = filteredFlowMag.get_log2();// mag.get_log2();
    logMag.get_histogram(1024).display_graph(0, 3);

    fstL.normalize(0.0f, 1.0f);

    CImgList<float> result(
            fstL,
            gradMag,
            flowMag,
            filteredFlowMag,
            logMag
            );
    result.display();
    */

    return 1;
}

