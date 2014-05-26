#pragma once

#include "common.h"

#include <Eigen/Dense>

#include "klt/klt.h"

class KLTTracker {
    public:
        KLTTracker();

        ~KLTTracker();

        void init(
                const CImg<uint8_t>& img,
                int numFeatures);

        void processNext(
                const CImg<uint8_t>& prev,
                const CImg<uint8_t>& img);

        void visualize(
                int frame,
                CImg<uint8_t>& img);

        inline const vector<vector<Eigen::Vector2f>>& getTracks() {
            return tracks;
        }
        
    private:
        KLT_TrackingContextRec *tCtx;
        KLT_FeatureListRec *curFL;

        vector<vector<Eigen::Vector2f>> tracks;
};

