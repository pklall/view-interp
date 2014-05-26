#pragma once

#include "common.h"

#include "daisy/daisy.h"

#include <nanoflann.hpp>

class DenseFeatureMatch {
    private:
        /*
        struct DaisyFlannAdapter {
            DenseFeatureMatch* match;

            inline size_t kdtree_get_point_count() const {
                // One descriptor point for each pixel
                return match->imgWidth * match->imgHeight;
            }

            inline void kdtree_get_pt(
                    const size_t idx,
                    float* result) const {
                int x = idx % match->imgWidth;
                int y = idx / match->imgWidth;
                int angle = match->angleAtXY(x, y);

                match->featureDetector.get_descriptor(y, x, angle, result);
            }

            template <class BBOX>
                bool kdtree_get_bbox(
                        BBOX &bb) const {
                    return false;
                }
        };

        typedef nanoflann::KDTreeSingleIndexAdaptor<
            L2_Adaptor_ArrPt<float, DaisyFlannAdapter, float, 200>,
            DaisyFlannAdapter,
            200> DaisyKDTree;
        */
    public:
        void match(
                const CImg<float>& imgGray);

    private:
        inline int angleAtXY(
                int x,
                int y) const {
            return 0;
        }

        int imgWidth, imgHeight;

        int angle;

        daisy featureDetector;

        // unique_ptr<DaisyKDTree> kdtree;
};
