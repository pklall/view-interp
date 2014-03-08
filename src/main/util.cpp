#include "common.h"

#include "util.h"

#include "opencv2/opencv.hpp"

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
            int corrY = my;

            float circleColor[3] = {255.0f, 0.0f, 0.0f};

            toAnnotated.draw_circle(corrX, corrY, 5, circleColor, 1.0f, 0);
        }

        toDisp.display(toAnnotated);
    }
}

void visualizeVectorImg(
        const CImg<float>& vec) {
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

void displayClickable(
        CImg<float>& img,
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
 * Computes optical flow from a to b and places the displacement field
 * into flow with x and y components in depth planes 0 and 1, respectively.
 *
 * FIXME this probably doesn't work
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

void dof(
    const CImg<float>& img,
    const CImg<float>& depth,
    float blurDepth,
    CImg<float>& processed) {

    CImg<float> I(5, 5);
    for (int i = 0; i < 3; i++) {
        cimg_forC(img, v) {
            int z = 0;
            cimg_for5x5(processed,x,y,z,v,I,float) {
                float blurSigma = 0.1f;
                float mouseDepth = blurDepth;
                if (depth(x, y) < mouseDepth) {
                    // FIXME
                    blurSigma = 1.5f * min(abs(
                            depth(x, y) - mouseDepth
                            ) / (150.0f), 1.0f);
                }
                float totalWeight = 0.0f;
                float result = 0.0f;
                for (int j = 0; j < 5; j++) {
                    for (int i = 0; i < 5; i++) {
                        float tx = i - 2;
                        float ty = j - 2;
                        float weight = exp(-(tx*tx + ty * ty) /
                                (blurSigma * blurSigma));
                        if (depth.atXY(x + tx, y + ty) >
                                depth.atXY(x, y)) {
                            weight = 0.0f;
                        }
                        totalWeight += weight;
                        result += weight * I(i, j);
                    }
                }
                processed(x, y, z, v) = result / totalWeight;
            }
        }
    }
}

void interactiveDOF(
        const CImg<float>& img,
        const CImg<float>& depth) {
    CImg<float> processed(img);
    CImgDisplay disp(processed);

    while (!disp.is_closed()) {
        disp.wait();

        float mx = disp.mouse_x() / (float) img.width();
        float my = disp.mouse_y() / (float) img.height();
        mx = min(mx, 1.0f);
        my = min(my, 1.0f);
        mx = max(mx, 0.0f);
        my = max(my, 0.0f);

        processed = img;

        float mouseDepth = depth.atXY(mx * (depth.width() - 1),
        my * (depth.height()- 1));

        dof(img, depth, mouseDepth, processed);


        disp.display(processed);
    }
}

void postProcessDepthMap(
        CImg<float>& labLeft,
        CImg<float>& leftField,
        CImg<float>& rightField) {
    const float INVALID = std::numeric_limits<float>::max();

    CImg<float> cost(leftField.width(), leftField.height());
    cost = 0.0f;
    cimg_forXY(leftField, x, y) {
        int rx = x + leftField(x, y, 0, 0);
        int ry = y;
        if (rx >= 0 && rx < rightField.width() &&
                ry >= 0 && ry < rightField.height()) {
            if (abs(rx - x + rightField(rx, ry)) > 2 ||
                    // FIXME this is a hack which only works on the 
                    // middleburry dataset
                    leftField(x, y) < 0) {
                cimg_forZC(leftField, z, c) {
                    leftField(x, y) = INVALID;
                }
                cost(x, y) = INVALID;
            }
        } else {
            cimg_forZC(leftField, z, c) {
                leftField(x, y) = INVALID;
            }
            cost(x, y) = INVALID;
        }
    }

    int dir = -1;
    for (int iter = 0; iter < 4; iter++) {
        dir *= -1;
        int startX = 0;
        int startY = 0;
        if (dir < 0) {
            startX = leftField.width() - 1;
            startY = leftField.height() - 1;
        }

        for (int y = startY; y >= 0 && y < leftField.height(); y += dir) {
            for (int x = startX; x >= 0 && x < leftField.width(); x += dir) {
                int neighborhood[2][2]{
                    {-dir, 0},
                    {0, -dir}
                };
                for (int i = 0; i < 2; i++) {
                    int nx = x + neighborhood[i][0];
                    int ny = y + neighborhood[i][1];
                    if (nx >= 0 && nx < leftField.width() &&
                            ny >= 0 && ny < leftField.height()) {
                        float edgeCost = 0;
                        cimg_forZC(labLeft, z, c) {
                            edgeCost +=
                                abs(labLeft(x, y, z, c) - labLeft(nx, ny, z, c));
                        }
                        if (cost(nx, ny) + edgeCost < cost(x, y)) {
                            cost(x, y) = cost(nx, ny) + edgeCost;
                            cimg_forZC(leftField, z, c) {
                                leftField(x, y, z, c) = leftField(nx, ny, z, c);
                            }
                        }
                    }
                }
            }
        }
    }
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
