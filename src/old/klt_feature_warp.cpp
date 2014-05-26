#include "klt_feature_warp.h"

#include "cvutil/cvutil.h"

KLTTracker::KLTTracker() :
    tCtx(nullptr),
    curFL(nullptr) {
}

KLTTracker::~KLTTracker() {
    if (tCtx != nullptr) {
        KLTFreeTrackingContext(tCtx);
    }

    if (curFL != nullptr) {
        KLTFreeFeatureList(curFL);
    }
}

void KLTTracker::init(
        const CImg<uint8_t>& img,
        int numFeatures) {
    tracks.clear();
    
    if (tCtx != nullptr) {
        KLTFreeTrackingContext(tCtx);
    }
    
    tCtx = KLTCreateTrackingContext();

    if (curFL != nullptr) {
        KLTFreeFeatureList(curFL);
    }

    curFL = KLTCreateFeatureList(numFeatures);

    KLTSelectGoodFeatures(tCtx, const_cast<uint8_t*>(img.data()),
            img.width(), img.height(), curFL);

    tracks.resize(curFL->nFeatures);

    for (int i = 0; i < curFL->nFeatures; i++) {
        const auto& f = curFL->feature[i];

        if (f->val >= 0) {
            tracks[i].push_back(Eigen::Vector2f(f->x, f->y));
        }
    }
}

void KLTTracker::processNext(
        const CImg<uint8_t>& prev,
        const CImg<uint8_t>& next) {
    KLTTrackFeatures(tCtx,
            const_cast<uint8_t*>(prev.data()),
            const_cast<uint8_t*>(next.data()),
            prev.width(), prev.height(),
            curFL);

    for (int i = 0; i < curFL->nFeatures; i++) {
        const auto& f = curFL->feature[i];
        
        if (f->val >= 0) {
            tracks[i].push_back(Eigen::Vector2f(f->x, f->y));
        }
    }
}

void KLTTracker::visualize(
        int frame,
        CImg<uint8_t>& img) {
    for (const vector<Eigen::Vector2f>& track : tracks) {
        if (track.size() > frame + 1) {
            const Eigen::Vector2f& cur = track[frame];
            const Eigen::Vector2f& next = track[frame + 1];

            uint8_t color[] = {255, 0, 0};

            img.draw_arrow(cur.x(), cur.y(), next.x(), next.y(), color, 30, 5);
        }
    }
}

