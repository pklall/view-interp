#pragma once

#include "common.h"

#include "segment.h"

#include <Eigen/Dense>
#include <Eigen/LU> 

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

struct AffineWarp {
    Eigen::AffineCompact2d transform;

    inline float* getParams() {
        return transform.data();
    }

    template<typename T>
    inline Eigen::Matrix<T, 2, 1> operator()(
            const Eigen::Matrix<T, 2, 1>& x) {
    }
};

// TODO solve for disparity per superpixel... then triangle superpixels
// through centers, and split along edges with depth discontinuities.

template<class FFIterator, class W, int WParamCount>
class WarpSolver {
    public:
        /**
         * Iterates over all samples of the flow field, providing samples
         * which may not be uniformly distributed over the image.
         *
         * Must implement the following functions:
         *  - void reset();
         *  - bool next();
         *  - tuple<Eigen::Vector2f, Eigen::Vector2f, int> operator*();
         *    Returns the (x, y, scale) associated with the current flow
         *    sample.
         */
        typedef FFI FlowFieldIterator;

        /**
         * Must implement the following functions:
         *  - float* getParams();
         *    Returns a pointer to an array of WParamCount floats specifying
         *    all of the parameters of the warp.
         *  - template<typename T>
         *    Eigen::Matrix<T, 2, 1> operator()(
         *        const Eigen::Matrix<T, 2, 1>& x);
         *    Evaluates the warp at the given point.
         */
        typedef W Warp;

        WarpSolver(
                const FlowFieldIterator* _flowField,
                const Segmentation* _segmentation,
                const Connectivity* _connectivity);

        void solve();

    private:
        const FlowFieldIterator* flowField;
        const Segmentation* segmentation;
        const Connectivity* connectivity;
};

template<class FFA, class W, int WParamCount>
WarpSolver::WarpSolver(
        const FlowFieldAdapter* _flowField,
        const Segmentation* _segmentation,
        const Connectivity* _connectivity) :
    flowField(_flowField),
    segmentation(_segmentation),
    connectivity(_connectivity) {
}

template<class FFA, class W, int WParamCount>
void WarpSolver::solve() {
    // TODO solve with ceres 
}

