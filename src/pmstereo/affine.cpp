#include "common.h"

#include "pmstereo.h"

#include <omp.h>

inline function<float(int, int, float[])> affinePatchDist(
        const CImg<float>& lab1,
        const CImg<float>& lab2,
        int wndSize,
        float colorSigma = 10.0f,
        float maxDist = 10.0f) {
    auto dist = [&lab1, &lab2, wndSize, colorSigma, maxDist]
        (int sx, int sy, float* value) -> float {
            float ssd = 0.0f;
            float totalWeight = 0.0f;

            for (int y = -wndSize / 2; y <= wndSize / 2; y++) {
                for (int x = -wndSize / 2; x <= wndSize / 2; x++) {
                    int srcX = x + sx;
                    int srcY = y + sy;

                    if (srcX >= 0 && srcX < lab1.width() &&
                            srcY >= 0 && srcY < lab1.height()) {

                        float dstX = srcX +
                            value[0] + value[1] * srcX + value[2] * srcY;
                        float dstY = srcY;

                        float lab1Diff = 0.0f;
                        cimg_forZC(lab1, z, c) {
                            float lDiff = lab1(srcX, srcY, z, c) -
                                lab1(sx, sy, z, c);
                            lab1Diff += abs(lDiff);
                        }
                        float weight = exp(-(lab1Diff) / colorSigma);

                        if (dstX >= 0 && dstX < lab2.width() &&
                                dstY >= 0 && dstY < lab2.height()) {
                            cimg_forZC(lab1, z, c) {
                                float diff = lab1(srcX, srcY, z, c) -
                                    lab2.linear_atXYZC(dstX, dstY, z, c);

                                diff = min(diff, maxDist);

                                ssd += abs(diff) * weight;
                                totalWeight += weight;
                            }
                        }
                    }
                }
            }

            if (totalWeight == 0.0f) {
                return std::numeric_limits<float>::max();
            }

            return ssd / totalWeight;
        };
    return dist;
}

