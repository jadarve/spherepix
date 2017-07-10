#include "spherepix/pixelation.h"

#include <iostream>
#include <algorithm>

#include "Eigen/Dense"

#include "spherepix/geometry.h"
#include "spherepix/springdynamics.h"

namespace spherepix {

//#########################################################
// FACE CONNECTIVITY GRAPHS
//#########################################################

/*
 * FIXME: Check new values according to new
 * origin of face coordinates at top corner.
 */
int __CONNECTIVITY_GRAPH_MODE_1[] = {3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0
                                    };

// TODO
int __CONNECTIVITY_GRAPH_MODE_2[] = {3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0,
                                     3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0
                                    };

// TODO
int __CONNECTIVITY_GRAPH_MODE_3[] = {3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0,
                                     3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0,
                                     3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0,
                                     3, 5, 1, 4,
                                     0, 5, 2, 4,
                                     1, 5, 3, 4,
                                     2, 5, 0, 4,
                                     3, 0, 1, 2,
                                     3, 2, 1, 0
                                    };

Image<int> FACE_CONNECTIVITY_MODE_1(6, 4, 1, &__CONNECTIVITY_GRAPH_MODE_1[0]);
Image<int> FACE_CONNECTIVITY_MODE_2(12, 4, 1, &__CONNECTIVITY_GRAPH_MODE_1[0]);
Image<int> FACE_CONNECTIVITY_MODE_3(24, 4, 1, &__CONNECTIVITY_GRAPH_MODE_1[0]);



//#########################################################
// MODULE METHODS
//#########################################################

float pixelSeparation(const Image<float>& etas) {

    int height = etas.height();
    int width = etas.width();

    Eigen::Vector3f eta = readVector3f(etas, height / 2, width / 2);
    Eigen::Vector3f eta_p = readVector3f(etas, height / 2, (width / 2) +1);

    Eigen::Vector3f mu = etaToMu_orthographic(eta, eta_p);
    return mu.norm();
}

Image<float> createFace_equidistant(const int N) {

    if ( N < 0) {
        std::cerr << "createFace_equidistant(): N must be greater than zero, got " << N << std::endl;
    }

    float a = cbrtf(3.0f) / 3.0f;
    float step = 2 * a / (N - 1);

    Image<float> face(N, N, 3);

    for (int row = 0; row < N; ++ row) {
        float y = -a + step * row;

        for (int col = 0; col < N; ++ col) {
            float x = -a + step * col;

            Eigen::Vector3f v = Eigen::Vector3f(x, -a, y).normalized();
            writeVector3f(v, row, col, face);
        }
    }

    return face;
}

Image<float> createFace_equiangular(const int N) {

    if ( N < 0) {
        std::cerr << "createFace_equiangular(): N must be greater than zero, got " << N << std::endl;
    }

    double a = cbrt(3.0f) / 3.0f;

    double vmin = -0.25 * M_PI;
    double step = 0.5 * M_PI / (N - 1);

    Image<float> face(N, N, 3);

    for (int row = 0; row < N; ++ row) {
        double y = a * tanf(vmin + step * row);

        for (int col = 0; col < N; ++ col) {
            double x = a * tanf(vmin + step * col);

            Eigen::Vector3f v = Eigen::Vector3f(x, -a, -y).normalized();
            writeVector3f(v, row, col, face);
        }
    }

    return face;
}

Image<float> regularizeCoordinates(const Image<float>& etas,
                                   const float Ls,
                                   const int I,
                                   const float dt,
                                   const float M,
                                   const float C,
                                   const float K) {

    int N = etas.height();
    Image<float> etas_in(etas.height(), etas.width(), 3);
    Image<float> etasVel_in(etas.height(), etas.width(), 3);
    Image<float> etasAcc_in(etas.height(), etas.width(), 3);

    etas_in.copyFrom(etas);
    etasVel_in.fill(0.0f);
    etasAcc_in.fill(0.0f);

    Image<float> etas_out(etas.height(), etas.width(), 3);
    Image<float> etasVel_out(etas.height(), etas.width(), 3);
    Image<float> etasAcc_out(etas.height(), etas.width(), 3);

    // float dt = 0.2f;
    // float M = 1.0f;
    // float C = 0.05f;
    // float K = 5.0f;

    Eigen::Vector3f eta_0 = readVector3f(etas, N / 2, N / 2);
    Eigen::Vector3f eta_1 = readVector3f(etas, N / 2, N / 2 + 1);

    Eigen::Vector3f mu = etaToMu_orthographic(eta_0, eta_1);
    float alpha = mu.norm();
    float L = alpha + (Ls * alpha / N);

    std::cout << "regularizeCoordinates(): alpha: " << alpha << std::endl;
    std::cout << "regularizeCoordinates(): L: " << L << std::endl;

    runSpringSystem(etas_in, etasVel_in, etasAcc_in,
                    etas_out, etasVel_out, etasAcc_out,
                    dt, M, C, K, L, I);

    return etas_out;
}

std::vector<Image<float> > createFace(PixelationMode mode,
                                      const int N,
                                      const int I,
                                      const float Ls,
                                      const float M3_theta,
                                      const float dt,
                                      const float M,
                                      const float C,
                                      const float K
                                     ) {

    std::vector<Image<float> > faceList;

    std::cout << "createFace(): N: " << N << std::endl;

    Image<float> baseFace = createFace_equiangular(N);


    switch (mode) {

    case MODE_1: {
        faceList.resize(1);
        Image<float> face1 = regularizeCoordinates(baseFace, Ls, I, dt, M, C, K);
        faceList[0] = face1;
        break;
    }

    case MODE_2: {
        std::cout << "createFace(): MODE_2: start" << std::endl;

        faceList.resize(2);

        int n = N / 2;
        Image<float> face1_in = baseFace.subImage(0, 0, n, baseFace.width());
        Image<float> face2_in = baseFace.subImage(n, 0, n, baseFace.width());

        Image<float> face1_reg = regularizeCoordinates(face1_in, Ls, I, dt, M, C, K);
        Image<float> face2_reg = regularizeCoordinates(face2_in, Ls, I, dt, M, C, K);

        faceList[0] = face1_reg;
        faceList[1] = face2_reg;

        std::cout << "createFace(): MODE_2: finished" << std::endl;
        break;
    }

    case MODE_3: {
        std::cout << "createFace(): MODE_3: start" << std::endl;

        faceList.resize(4);

        int n = N / 2;
        std::cout << "n: " << n << std::endl;

        Image<float> face1_reg = regularizeCoordinates(baseFace.subImage(0, 0, n, n), Ls, I, dt, M, C, K);
        Image<float> face2_reg = regularizeCoordinates(baseFace.subImage(0, n, n, n), Ls, I, dt, M, C, K);
        Image<float> face3_reg = regularizeCoordinates(baseFace.subImage(n, 0, n, n), Ls, I, dt, M, C, K);
        Image<float> face4_reg = regularizeCoordinates(baseFace.subImage(n, n, n, n), Ls, I, dt, M, C, K);

        // reads the middle coordinate of the face and use if as normal vector of the face
        Eigen::Vector3f N1 = readVector3f(face1_reg, n / 2, n / 2);
        Eigen::Vector3f N2 = readVector3f(face2_reg, n / 2, n / 2);
        Eigen::Vector3f N3 = readVector3f(face3_reg, n / 2, n / 2);
        Eigen::Vector3f N4 = readVector3f(face4_reg, n / 2, n / 2);

        // Rotation axis
        Eigen::Vector3f Raxis_1 = N1.cross(N4).normalized();
        Eigen::Vector3f Raxis_2 = N2.cross(N3).normalized();

        // Rotation matrices
        Eigen::Matrix3f R_1 = rotationMatrixAxisAngle(Raxis_1, -M3_theta);
        Eigen::Matrix3f R_4 = rotationMatrixAxisAngle(Raxis_1, M3_theta);

        Eigen::Matrix3f R_2 = rotationMatrixAxisAngle(Raxis_2, -M3_theta);
        Eigen::Matrix3f R_3 = rotationMatrixAxisAngle(Raxis_2, M3_theta);

        // Apply the rotation transform to the regularized coordinates
        // overwriting the original values
        transformCoordinates(face1_reg, R_1, face1_reg);
        transformCoordinates(face2_reg, R_2, face2_reg);
        transformCoordinates(face3_reg, R_3, face3_reg);
        transformCoordinates(face4_reg, R_4, face4_reg);


        faceList[0] = face1_reg;
        faceList[1] = face2_reg;
        faceList[2] = face3_reg;
        faceList[3] = face4_reg;

        std::cout << "createFace(): MODE_3: finished" << std::endl;
        break;
    }

    }

    return faceList;
}

std::vector<Image<float> > createCubeFaces(std::vector<Image<float> >& face) {

    std::cout << "createCubeFaces(): start" << std::endl;
    std::vector<Image<float> > cubeFaces;

    // rotation matrices for making the cube faces
    std::vector<Eigen::Matrix3f> rotationList {
        rotationMatrixAxisAngle(Eigen::Vector3f::UnitZ(), 0.0f * M_PI),
        rotationMatrixAxisAngle(Eigen::Vector3f::UnitZ(), 0.5f * M_PI),
        rotationMatrixAxisAngle(Eigen::Vector3f::UnitZ(), 1.0f * M_PI),
        rotationMatrixAxisAngle(Eigen::Vector3f::UnitZ(), 1.5f * M_PI),
        rotationMatrixAxisAngle(Eigen::Vector3f::UnitX(), 0.5f * M_PI),     // south
        rotationMatrixAxisAngle(Eigen::Vector3f::UnitX(), -0.5f * M_PI)     // north
    };

    for (auto R : rotationList) {
        for (auto etas : face) {
            cubeFaces.push_back(transformCoordinates(etas, R));
        }
    }

    std::cout << "createCubeFaces(): finished" << std::endl;

    return cubeFaces;
}


const Image<int> getFaceConnectivityGraph(PixelationMode mode) {

    switch (mode) {

    case MODE_1:
        return FACE_CONNECTIVITY_MODE_1;

    case MODE_2:
        return FACE_CONNECTIVITY_MODE_2;

    case MODE_3:
        return FACE_CONNECTIVITY_MODE_3;

    default:
        std::cerr << "getFaceConnectivityGraph(): ERROR: unexpected pixelation mode: " << mode << std::endl;
        return FACE_CONNECTIVITY_MODE_3;
    }
}


std::vector<std::pair<Image<float>, Image<float>>> faceInterpolationBelts(std::vector<Image<float>>& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode) {

    std::vector<std::pair<Image<float>, Image<float>>> tupleList;

    tupleList.push_back(interpolationBelt_0(faceList, faceIndex, beltWidth, mode));
    tupleList.push_back(interpolationBelt_1(faceList, faceIndex, beltWidth, mode));
    tupleList.push_back(interpolationBelt_2(faceList, faceIndex, beltWidth, mode));
    tupleList.push_back(interpolationBelt_3(faceList, faceIndex, beltWidth, mode));

    return tupleList;
}

std::pair<Image<float>, Image<float>> interpolationBelt_0(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode) {

    // gets the face connectivity graph
    Image<int> faceConnGraph = getFaceConnectivityGraph(mode);

    Image<float> face = faceList[faceIndex];
    Image<float> betas(face.height(), beltWidth, 2);
    Image<float> etas(face.height(), beltWidth, 3);
    Image<float> neighborFace = faceList[faceConnGraph(faceIndex, 0)];

    for (int r = 0; r < betas.height(); ++ r) {

        Eigen::Vector3f eta = readVector3f(face, r, 0);

        // Orthonormal basis
        float norm = 0.0f;
        Eigen::Matrix2Xf B = getOrthonormalBasis(face, r, 0, norm);

        for (int c = 0; c < betas.width(); ++ c) {

            // pixel coordinate (col, row)
            Eigen::Vector2f beta(c - beltWidth, 0.0f);

            // interpolated spherical coordinate
            Eigen::Vector3f muCol = (B.transpose() * beta) * norm;
            Eigen::Vector3f etaCol = muToEta_orthographic(eta, muCol);

            // search interpolated coordinates (row, col)
            Eigen::Vector2f betaInterp = findInterpolationCoodinates(etaCol, neighborFace);

            // write ourput
            writeVector2f(betaInterp, r, c, betas);
            writeVector3f(etaCol, r, c, etas);

            // std::cout << "interpolationBelt_0(): [" << r << ", " << c << "]" << std::endl;
            // std::cout << "B:\n" << B << std::endl;
            // std::cout << "norm: " << norm << std::endl;
            // std::cout << "beta: " << beta.transpose() << std::endl;
            // std::cout << "muCol: " << muCol.transpose() << std::endl;
            // std::cout << "etaCol: " << etaCol.transpose() << std::endl;
            // std::cout << "betaInterp: " << betaInterp.transpose() << std::endl;
        }
    }

    return std::pair<Image<float>, Image<float>>(betas, etas);
}

std::pair<Image<float>, Image<float>> interpolationBelt_1(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode) {

    // gets the face connectivity graph
    Image<int> faceConnGraph = getFaceConnectivityGraph(mode);

    Image<float> face = faceList[faceIndex];
    Image<float> betas(beltWidth, face.width() + 2 * beltWidth, 2);
    Image<float> etas(beltWidth, face.width() + 2 * beltWidth, 3);
    Image<float> neighborFace = faceList[faceConnGraph(faceIndex, 1)];

    // row coordinate of base vector eta
    int baseRow = 0;

    for (int r = 0; r < betas.height(); ++ r) {

        for (int c = 0; c < betas.width(); ++ c) {

            // column coordinate of the base vector eta
            int cc = c - beltWidth;
            int baseCol = std::max(0, std::min(cc, face.width() - 1));

            // base vector
            Eigen::Vector3f eta = readVector3f(face, baseRow, baseCol);

            // Orthonormal basis
            float norm = 0.0f;
            Eigen::Matrix2Xf B = getOrthonormalBasis(face, baseRow, baseCol, norm);

            // pixel coordinate (col, row)
            int betaCol = cc < 0 ? cc : c > face.width() + beltWidth - 1 ? c - face.width() - beltWidth + 1 : 0;
            Eigen::Vector2f beta(betaCol, r - beltWidth);

            // interpolated spherical coordinate
            Eigen::Vector3f muCol = (B.transpose() * beta) * norm;
            Eigen::Vector3f etaCol = muToEta_orthographic(eta, muCol);

            // search interpolated coordinates (row, col)
            Eigen::Vector2f betaInterp = findInterpolationCoodinates(etaCol, neighborFace);

            // write ourput
            writeVector2f(betaInterp, r, c, betas);
            writeVector3f(etaCol, r, c, etas);

            // std::cout << "interpolationBelt_0(): [" << r << ", " << c << "]" << std::endl;
            // std::cout << "base coord: [" << baseRow << ", " << baseCol << "]" << std::endl;
            // std::cout << "B:\n" << B << std::endl;
            // std::cout << "norm: " << norm << std::endl;
            // std::cout << "beta: " << beta.transpose() << std::endl;
            // std::cout << "muCol: " << muCol.transpose() << std::endl;
            // std::cout << "etaCol: " << etaCol.transpose() << std::endl;
            // std::cout << "betaInterp: " << betaInterp.transpose() << std::endl;
        }
    }

    return std::pair<Image<float>, Image<float>>(betas, etas);
}

std::pair<Image<float>, Image<float>> interpolationBelt_2(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode) {

    // gets the face connectivity graph
    Image<int> faceConnGraph = getFaceConnectivityGraph(mode);

    Image<float> face = faceList[faceIndex];
    Image<float> betas(face.height(), beltWidth, 2);
    Image<float> etas(face.height(), beltWidth, 3);
    Image<float> neighborFace = faceList[faceConnGraph(faceIndex, 2)];

    for (int r = 0; r < betas.height(); ++ r) {

        int baseRow = r;
        int baseCol = face.width() - 1;

        Eigen::Vector3f eta = readVector3f(face, baseRow, baseCol);

        // Orthonormal basis
        float norm = 0.0f;
        Eigen::Matrix2Xf B = getOrthonormalBasis(face, baseRow, baseCol, norm);

        for (int c = 0; c < betas.width(); ++ c) {

            // pixel coordinate (col, row)
            Eigen::Vector2f beta(c + 1, 0.0f);

            // interpolated spherical coordinate
            Eigen::Vector3f muCol = (B.transpose() * beta) * norm;
            Eigen::Vector3f etaCol = muToEta_orthographic(eta, muCol);

            // search interpolated coordinates (col, row)
            Eigen::Vector2f betaInterp = findInterpolationCoodinates(etaCol, neighborFace);

            // write ourput
            writeVector2f(betaInterp, r, c, betas);
            writeVector3f(etaCol, r, c, etas);

            // std::cout << "interpolationBelt_0(): [" << r << ", " << c << "]" << std::endl;
            // std::cout << "B:\n" << B << std::endl;
            // std::cout << "norm: " << norm << std::endl;
            // std::cout << "beta: " << beta.transpose() << std::endl;
            // std::cout << "muCol: " << muCol.transpose() << std::endl;
            // std::cout << "etaCol: " << etaCol.transpose() << std::endl;
            // std::cout << "betaInterp: " << betaInterp.transpose() << std::endl;
        }
    }

    return std::pair<Image<float>, Image<float>>(betas, etas);
}

std::pair<Image<float>, Image<float>> interpolationBelt_3(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode) {

    // gets the face connectivity graph
    Image<int> faceConnGraph = getFaceConnectivityGraph(mode);

    Image<float> face = faceList[faceIndex];
    Image<float> betas(beltWidth, face.width() + 2 * beltWidth, 2);
    Image<float> etas(beltWidth, face.width() + 2 * beltWidth, 3);
    Image<float> neighborFace = faceList[faceConnGraph(faceIndex, 3)];

    // row coordinate of base vector eta
    int baseRow = face.height() - 1;

    for (int r = 0; r < betas.height(); ++ r) {

        for (int c = 0; c < betas.width(); ++ c) {

            // column coordinate of the base vector eta
            int cc = c - beltWidth;
            int baseCol = std::max(0, std::min(cc, face.width() - 1));

            // base vector
            Eigen::Vector3f eta = readVector3f(face, baseRow, baseCol);

            // Orthonormal basis
            float norm = 0.0f;
            Eigen::Matrix2Xf B = getOrthonormalBasis(face, baseRow, baseCol, norm);

            // pixel coordinate (col, row)
            int betaCol = cc < 0 ? cc : c > face.width() + beltWidth - 1 ? c - face.width() - beltWidth + 1 : 0;
            Eigen::Vector2f beta(betaCol, r + 1);

            // interpolated spherical coordinate
            Eigen::Vector3f muCol = (B.transpose() * beta) * norm;
            Eigen::Vector3f etaCol = muToEta_orthographic(eta, muCol);

            // search interpolated coordinates (row, col)
            Eigen::Vector2f betaInterp = findInterpolationCoodinates(etaCol, neighborFace);

            // write ourput
            writeVector2f(betaInterp, r, c, betas);
            writeVector3f(etaCol, r, c, etas);

            // std::cout << "interpolationBelt_0(): [" << r << ", " << c << "]" << std::endl;
            // std::cout << "base coord: [" << baseRow << ", " << baseCol << "]" << std::endl;
            // std::cout << "B:\n" << B << std::endl;
            // std::cout << "norm: " << norm << std::endl;
            // std::cout << "beta: " << beta.transpose() << std::endl;
            // std::cout << "muCol: " << muCol.transpose() << std::endl;
            // std::cout << "etaCol: " << etaCol.transpose() << std::endl;
            // std::cout << "betaInterp: " << betaInterp.transpose() << std::endl;
        }
    }

    return std::pair<Image<float>, Image<float>>(betas, etas);
}


//#########################################################
// Pixelation
//#########################################################

Pixelation::Pixelation() {

}

Pixelation::Pixelation(PixelationMode mode, const std::vector<Image<float>>& faceCoordinates,
                       const std::vector<std::vector<Image<float>>>& betasInterpolation,
                       const std::vector<std::vector<Image<float>>>& etasInterpolation,
                       const int interpolationBeltWidth) {

    __mode = mode;
    __beltWidth = interpolationBeltWidth;
    __faceConnectivityGraph = getFaceConnectivityGraph(mode);

    __coordinates = faceCoordinates;
    __betasBeltCoordinates = betasInterpolation;
    __etasBeltCoordinates = etasInterpolation;

    __faceHeight = faceCoordinates[0].height();
    __faceWidth = faceCoordinates[0].width();
}

Pixelation::~Pixelation() {
    // nothing to do
}

PixelationMode Pixelation::mode() const {
    return __mode;
}

int Pixelation::faceHeight() const {
    return __faceHeight;
}

int Pixelation::faceWidth() const {
    return __faceWidth;
}

int Pixelation::interpolationBeltWidth() const {
    return __beltWidth;
}

int Pixelation::faceCount() const {
    return __coordinates.size();
}

const Image<int> Pixelation::faceConnectivityGraph() const {
    return __faceConnectivityGraph;
}

const Image<float> Pixelation::faceCoordinates(const int faceIndex) {
    return __coordinates[faceIndex];
}

const Image<float> Pixelation::interpolationCoordinates(const int faceIndex,
        FaceNeighbor neighbor) const {

    return __betasBeltCoordinates[faceIndex][neighbor];
}

const Image<float> Pixelation::sphericalInterpolationCoordinates(const int faceIndex,
        FaceNeighbor neighbor) const {
    return __etasBeltCoordinates[faceIndex][neighbor];
}



//#########################################################
// PixelationFactory
//#########################################################

Pixelation PixelationFactory::createPixelation(PixelationMode mode,
        const int N,
        const int interpolationBeltWidth,
        const int springIterations,
        const float extraSeparation,
        const float M3_theta,
        const float dt,
        const float M,
        const float C,
        const float K) {

    // base pixelation base
    std::vector<Image<float>> etasBase = createFace(mode, N, springIterations,
                                         extraSeparation, M3_theta, dt, M, C, K);

    return createPixelation(mode, etasBase, interpolationBeltWidth);
}

Pixelation PixelationFactory::createPixelation(PixelationMode mode,
                                               std::vector<Image<float>>& faceList,
                                               const int interpolationBeltWidth) {

    // creates the six faces that make the pixelation
    std::vector<Image<float>> cubeFaces = createCubeFaces(faceList);

    // interpolation belts for each face
    std::vector<std::vector<Image<float>>> betaCoordinates;
    std::vector<std::vector<Image<float>>> etaInterpolationCoordinates;

    for (unsigned int faceIndex = 0; faceIndex < cubeFaces.size(); faceIndex ++) {

        std::vector<std::pair<Image<float>, Image<float>>> beltCoords =
            faceInterpolationBelts(cubeFaces, faceIndex, interpolationBeltWidth, mode);

        std::vector<Image<float>> betas;
        std::vector<Image<float>> etas;

        for (auto c : beltCoords) {
            betas.push_back(c.first);
            etas.push_back(c.second);
        }

        betaCoordinates.push_back(betas);
        etaInterpolationCoordinates.push_back(etas);
    }

    // creates Pixelation object
    Pixelation pix(mode, cubeFaces, betaCoordinates,
                   etaInterpolationCoordinates, interpolationBeltWidth);

    return pix;
}

Pixelation PixelationFactory::load(const std::string& path) {
    // TODO
    return Pixelation();
}

void PixelationFactory::save(const Pixelation& pix) {
    // TODO
}


//#########################################################
// COORDINATE CASTING
//#########################################################

Image<float> castCoordinates(const Image<float>& etas_0,
    const Image<float>& etas_1, const bool flipVertical) {

    Image<float> betas(etas_0.height(), etas_0.width(), 2);
    betas.clear();

    for(int r = 0; r < etas_0.height(); r ++) {
        for(int c = 0; c < etas_0.width(); c ++) {

            Eigen::Vector3f eta = readVector3f(etas_0, r, c);

            Eigen::Vector2f beta = findInterpolationCoodinates(eta, etas_1, flipVertical);

            // col and row coordinates
            float br = beta.y();
            float bc = beta.x();

            if(br >= 0.0f && br < etas_1.height() && bc >= 0 && bc < etas_1.width()) {
                writeVector2f(beta, r, c, betas);
            } else {
                writeVector2f(Eigen::Vector2f(-1.0f, -1.0f), r, c, betas);
            }
        }
    }

    return betas;
}

}; // namespace spherepix