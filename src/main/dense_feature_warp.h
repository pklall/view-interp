#pragma once

#include "common.h"

#include "daisy/daisy.h"

#include <nanoflann.hpp>

template<class T, class DataSource, typename _DistanceType, int SIZE>
struct L2_Adaptor_ArrPt {
    typedef T ElementType;
    typedef _DistanceType DistanceType;

    const DataSource &data_source;

    L2_Adaptor_ArrPt(const DataSource &_data_source) : data_source(_data_source) { }

    inline DistanceType operator()(const T* a, const size_t b_idx, size_t size, DistanceType worst_dist = -1) const
    {
        DistanceType result = DistanceType();
        const T* last = a + size;
        const T* lastgroup = last - 3;
        size_t d = 0;

        T b[SIZE];

        data_source.kdtree_get_pt(b_idx, b);

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup) {
            const DistanceType diff0 = a[0] - b[d++];
            const DistanceType diff1 = a[1] - b[d++];
            const DistanceType diff2 = a[2] - b[d++];
            const DistanceType diff3 = a[3] - b[d++];
            result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4;
            if ((worst_dist>0)&&(result>worst_dist)) {
                return result;
            }
        }
        /* Process last 0-3 components.  Not needed for standard vector lengths. */
        while (a < last) {
            const DistanceType diff0 = *a++ - b[d++];
            result += diff0 * diff0;
        }
        return result;
    }

    template <typename U, typename V>
        inline DistanceType accum_dist(const U a, const V b, int ) const
        {
            return (a-b)*(a-b);
        }
};

class DenseFeatureMatch {
    private:
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

        unique_ptr<DaisyKDTree> kdtree;
};
