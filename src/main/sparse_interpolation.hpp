#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * Sparse data interpolation over uniform grids supporting MRF priors.
 */
template<typename S>
class SparseInterp {
    public:
        typedef S Scalar;

        SparseInterp(
                int _width,
                int _height) :
            width(_width),
            height(_height) {
            X.resize(width * height);
        }

        void init(
                int numSamples,
                Scalar pairwiseWeight) {
            A.resize(width * height + numSamples, width * height);
            // Expect at most 6 non-zero elements in each column
            A.reserve(Eigen::VectorXi::Constant(width * height, 6));

            B.resize(width * height + numSamples);

            int neighborhood[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

            // initialize laplacian prior
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = coordToIndex(x, y);

                    int neighborCount = 0;

                    for (int n = 0; n < 4; n++) {
                        int nx = x + neighborhood[n][0];
                        int ny = y + neighborhood[n][1];

                        if (
                                nx >= 0 ||
                                nx < width ||
                                ny >= 0 ||
                                ny < height) {
                            neighborCount++;
                            A(idx, coordToIndex(nx, ny)) = Scalar(1);
                        }
                    }

                    A(idx, idx) = -neighborCount;

                    B(idx) = Scalar(0);
                }
            }
        }

        inline void insertSample(
            int sampleIndex,
            int x,
            int y,
            Scalar val) {
            A(width * height + sampleIndex, coordToIndex(x, y)) = Scalar(1);

            B(width * height + sampleIndex) = val;
        }

        void solve() {
            Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>> cg;
            cg.compute(A);
            X = cg.solve(B);
            // std::cout << "#iterations:     " << cg.iterations() << std::endl;
            // std::cout << "estimated error: " << cg.error()      << std::endl;
        }

        inline const Scalar& solutionAt(
                int x,
                int y) {
            return X(coordToIndex(x, y));
        }

    private:
        inline int coordToIndex(
                int x,
                int y) {
            return x + y * width;
        }

        const int width;
        const int height;

        Eigen::SparseMatrix<Scalar> A;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> B;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> X;
};

