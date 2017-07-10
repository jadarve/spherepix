#include "spherepix/geometry.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

#include "Eigen/Geometry"
#include "spherepix/arithmetic.h"

namespace spherepix {

/////////////////////////////////////////////////
// TANGENT SPACE PROJECTION MATRIX
/////////////////////////////////////////////////
Eigen::Matrix3f projectionMatrix(const Eigen::Vector3f& eta) {
    return Eigen::Matrix3f::Identity() - eta * eta.transpose();
}

/////////////////////////////////////////////////
// RETRACTIONS
/////////////////////////////////////////////////

Eigen::Vector3f etaToMu_orthographic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta) {
    return projectionMatrix(eta_0) * eta;
}

Eigen::Vector3f etaToMu_perspective(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta) {
    float alpha = 1.0f / (eta_0.dot(eta));
    return alpha * eta - eta_0;
}

Eigen::Vector3f etaToMu_geodesic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta) {

    Eigen::Vector3f muOrtho = etaToMu_orthographic(eta_0, eta);
    float muNorm = muOrtho.norm();
    if (muNorm > 0.0f) {
        float theta = acosf(eta_0.dot(eta));
        theta = isnan(theta) ? 0.0f : theta;
        return muOrtho * theta / muNorm;
    } else {
        // std::cout << "etaToMu_geodesic(): returning zero: " << muNorm << std::endl;
        return Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    }

    // WARNING: for the case eta == -eta_0 (eta is the inverse of eta),
    //  dotProduct = -1 and the retraction is undefined. Notice that it is
    //  not possible to recover an unique path from eta_0 to eta.
}

Eigen::Vector3f etaToMu_cordal(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta) {
    Eigen::Vector3f nu = etaToMu_orthographic(eta_0, eta).normalized();
    return nu * (eta_0 - eta).norm();
}


/////////////////////////////////////////////////
// INVERSE RETRACTIONS
/////////////////////////////////////////////////

Eigen::Vector3f muToEta_orthographic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu) {
    return mu + eta_0 * sqrtf(1.0f - mu.dot(mu));
}

Eigen::Vector3f muToEta_perspective(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu) {
    Eigen::Vector3f eta = eta_0 + mu;
    return eta.normalized();
}

Eigen::Vector3f muToEta_geodesic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu) {

    float muNorm = mu.norm();
    if (isZero(muNorm)) {
        return Eigen::Vector3f(eta_0);
    }

    // compute rotation axis
    Eigen::Vector3f axis = eta_0.cross(mu).normalized();

    // rotation matrix
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(muNorm, axis);
    return R * eta_0;
}

Eigen::Vector3f muToEta_cordal(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu) {

    float muNorm = mu.norm();

    if (isZero(muNorm)) {
        return Eigen::Vector3f(eta_0);
    }

    // rotation axis
    Eigen::Vector3f axis = eta_0.cross(mu).normalized();
    float theta = 2 * asinf(0.5f * muNorm);

    // rotation matrix
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(theta, axis);

    return R * eta_0;
}


/////////////////////////////////////////////////
// ROTATION MATRICES
/////////////////////////////////////////////////

Eigen::Matrix3f rotationMatrix(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta_1) {

    float theta = acosf(eta_0.dot(eta_1));
    // theta = isnan(theta) ? 0.0f : theta;

    if (fabs(theta) > 0.0f) {

        Eigen::Vector3f axis = eta_0.cross(eta_1);
        axis.normalize();
        if(isNaN(axis)) {
            std::cout << "rMatrix: " << isNaN(eta_0) << ", " << isNaN(eta_1) << std::endl;
            return Matrix3f::Identity();
        }

        Eigen::Matrix3f axisCross;
        axisCross << 0.0f, -axis.z(), axis.y(),
                  axis.z(), 0.0f, -axis.x(),
                  -axis.y(), axis.x(), 0.0f;

        float xx = axis.x() * axis.x();
        float xy = axis.x() * axis.y();
        float xz = axis.x() * axis.z();

        float yy = axis.y() * axis.y();
        float yz = axis.y() * axis.z();

        float zz = axis.z() * axis.z();

        Eigen::Matrix3f tensorMatrix;
        tensorMatrix << xx, xy, xz,
                     xy, yy, yz,
                     xz, yz, zz;

        Eigen::Matrix3f R = cosf(theta) * Eigen::Matrix3f::Identity() +
                            sinf(theta) * axisCross + (1 - cosf(theta)) * tensorMatrix;

        return R;
    } else {
        return Matrix3f::Identity();
    }
}

Eigen::Matrix3f rotationMatrixAxisAngle(const Eigen::Vector3f& axis, const float& theta) {

    // rotation matrix
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(theta, axis);
    return R;
}


/////////////////////////////////////////////////
// ORTHONORMAL COORDINATES
/////////////////////////////////////////////////

Eigen::Matrix2Xf getOrthonormalBasis(const Image<float>& etas, const int row, const int col) {

    // read base vector from the grid
    Eigen::Vector3f eta_0 = readVector3f(etas, row, col);

    // offset sign for reading eta from grid
    int sign = col < etas.width() - 1 ? 1 : -1;

    Eigen::Vector3f eta_col = readVector3f(etas, row, col + sign);

    // if sign is negative, eta is read from col - 1, and mu needs to be mirrowed
    Eigen::Vector3f mu_col = sign * etaToMu_orthographic(eta_0, eta_col);
    
    // Eigen::Vector3f mu_row = eta_0.cross(mu_col);
    Eigen::Vector3f mu_row = mu_col.cross(eta_0);

    mu_col.normalize();
    mu_row.normalize();

    Eigen::Matrix2Xf B(2, 3);
    B << mu_col.transpose(), mu_row.transpose();

    return B;
}


Eigen::Matrix2Xf getOrthonormalBasis(const Image<float>& etas, const int row, const int col, float& norm_out) {

    // read base vector from the grid
    Eigen::Vector3f eta_0 = readVector3f(etas, row, col);

    // offset sign for reading eta from grid
    int sign = col < etas.width() - 1 ? 1 : -1;

    Eigen::Vector3f eta_col = readVector3f(etas, row, col + sign);

    // if sign is negative, eta is read from col - 1, and mu needs to be mirrowed
    Eigen::Vector3f mu_col = sign * etaToMu_orthographic(eta_0, eta_col);
    // Eigen::Vector3f mu_row = eta_0.cross(mu_col);
    Eigen::Vector3f mu_row = mu_col.cross(eta_0);

    norm_out = mu_col.norm();
    mu_col.normalize();
    mu_row.normalize();

    Eigen::Matrix2Xf B(2, 3);
    B << mu_col.transpose(), mu_row.transpose();

    return B;
}


// Eigen::Matrix2Xf getOrthonormalBasis(const Image<float>& etas, const int row, const int col, float& norm_out) {

//     // read base vector from the grid
//     Eigen::Vector3f eta_0 = readVector3f(etas, row, col);

//     // offset sign for reading eta from grid
//     int sign_c = col < etas.width() - 1 ? 1 : -1;
//     int sign_r = row < etas.height() - 1 ? 1 : -1;

//     Eigen::Vector3f eta_col = readVector3f(etas, row, col + sign_c);
//     Eigen::Vector3f eta_row = readVector3f(etas, row + sign_r, col);

//     // if sign is negative, eta is read from col - 1, and mu needs to be mirrowed
//     Eigen::Vector3f mu_col = sign_c * etaToMu_orthographic(eta_0, eta_col);
//     Eigen::Vector3f mu_row = sign_r * etaToMu_orthographic(eta_0, eta_row);

//     norm_out = mu_col.norm();
//     mu_col.normalize();
//     mu_row.normalize();

//     Eigen::Matrix2Xf B(2, 3);
//     B << mu_col.transpose(), mu_row.transpose();

//     return B;
// }


Eigen::Vector2f muToBetapix(const Eigen::Matrix2Xf& B, const float Bnorm, const Eigen::Vector3f& mu) {
    return B * mu / Bnorm;
}


Eigen::Vector3f betapixToMu(const Eigen::Matrix2Xf& B, const float Bnorm, const Eigen::Vector2f& beta) {
    return B.transpose() * beta * Bnorm;
}


/////////////////////////////////////////////////
// COORDINATES INTERPOLATION
/////////////////////////////////////////////////

Eigen::Vector2f findInterpolationCoodinates(const Eigen::Vector3f& eta,
    const Image<float>& etaGrid, const bool flipVertical) {

    const bool checkCoordinates = true;

    // center of the 2D search space
    int centerH = (etaGrid.height() / 2) - 1;
    int centerW = (etaGrid.width() / 2) - 1;

    // increment
    int incH = etaGrid.height() / 4;
    int incW = etaGrid.width() / 4;

    // children index in the subdivision
    // int childrenIndex = 0;

    // QuadTree search algorithm
    while (incH > 0 || incW > 0) {

        // std::cout << "(cH, cW): (" << centerH << ", " << centerW
        //           << ")\t(iH, iW): (" << incH << ", " << incW << ")\tchild: " << childrenIndex << std::endl;

        Eigen::Vector3f eta_00 = readVector3f<checkCoordinates>(etaGrid, centerH - incH, centerW - incW);
        Eigen::Vector3f eta_01 = readVector3f<checkCoordinates>(etaGrid, centerH - incH, centerW + incW);
        Eigen::Vector3f eta_10 = readVector3f<checkCoordinates>(etaGrid, centerH + incH, centerW - incW);
        Eigen::Vector3f eta_11 = readVector3f<checkCoordinates>(etaGrid, centerH + incH, centerW + incW);

        float n_00 = etaToMu_geodesic(eta_00, eta).norm();
        float n_01 = etaToMu_geodesic(eta_01, eta).norm();
        float n_10 = etaToMu_geodesic(eta_10, eta).norm();
        float n_11 = etaToMu_geodesic(eta_11, eta).norm();

        // find the index of minimum norm
        std::vector<float> norm_vector = {n_00, n_01, n_10, n_11};
        int childrenIndex = std::min_element(norm_vector.begin(), norm_vector.end()) - norm_vector.begin();

        // assign the new center position according to the children index
        // with minimum distance
        switch (childrenIndex) {
        case 0:
            centerH = centerH - incH;
            centerW = centerW - incW;
            break;
        case 1:
            centerH = centerH - incH;
            centerW = centerW + incW;
            break;
        case 2:
            centerH = centerH + incH;
            centerW = centerW - incW;
            break;
        case 3:
            centerH = centerH + incH;
            centerW = centerW + incW;
            break;
        }

        incH /= 2;
        incW /= 2;
    }

    // std::cout << "-----------------------------------------------------------------" << std::endl;
    // std::cout << "(cH, cW): (" << centerH << ", " << centerW
    //           << ")\t(iH, iW): (" << incH << ", " << incW << ")\tchild: " << childrenIndex << std::endl;

    // the last centerH/W is the nearest pixel position to eta
    Eigen::Vector3f eta_0 = readVector3f<checkCoordinates>(etaGrid, centerH, centerW);

    // get the interpolation basis
    float Bnorm = 0.0f;
    Eigen::Matrix2Xf B = getOrthonormalBasis(etaGrid, centerH, centerW, Bnorm);

    Eigen::Vector3f mu = etaToMu_geodesic(eta_0, eta);

    // interpolated beta coordinates
    // Eigen::Vector2f betaInterp = (B * mu) / Bnorm;
    Eigen::Vector2f betaInterp = muToBetapix(B, Bnorm, mu);

    // (col, row) coordinates
    // betaInterp.x() -> col coordinate
    // betaInterp.y() -> row coordinate

    // Eigen::Vector2f beta((float)centerW + betaInterp.x(), (float)centerH + betaInterp.y());

    if(flipVertical) {
        return Eigen::Vector2f((float)centerW + betaInterp.x(), (float)centerH + betaInterp.y());
    } else {
        return Eigen::Vector2f((float)centerW + betaInterp.x(), (float)centerH - betaInterp.y());
    }

    // std::cout << "eta_0:\t\t" << eta_0.transpose() << std::endl;
    // std::cout << "eta:\t\t" << eta.transpose() << std::endl;
    // std::cout << "B:\n" << B << std::endl;
    // std::cout << "Bnorm: " << Bnorm << std::endl;
    // std::cout << "mu interp:\t" << mu.transpose() << std::endl;
    // std::cout << "beta interp:\t" << betaInterp.transpose() << std::endl;
    // std::cout << "beta:\t\t" << beta.transpose() << std::endl;

    // return beta;
}

/////////////////////////////////////////////////
// UTILITY FUNCTIONS
/////////////////////////////////////////////////

void transformCoordinates(const Image<float>& etas, const Eigen::Matrix3f& T, Image<float>& etas_out) {

    if (etas.depth() != 3) {
        std::cerr << "transformCoordinates(): ERROR: etas grid must have depth 3,  got " << etas.depth() << std::endl;
    }

    if (!etas.compareShape(etas_out)) {
        std::cerr << "transformCoordinates(): ERROR: etas_out must have same shape than input grid" << std::endl;
    }

    for (int r = 0; r < etas.height(); ++ r) {
        for (int c = 0; c < etas.width(); ++c ) {

            Eigen::Vector3f v = readVector3f(etas, r, c);
            writeVector3f(T * v, r, c, etas_out);
        }
    }
}

Image<float> transformCoordinates(const Image<float>& etas, const Eigen::Matrix3f& T) {

    if (etas.depth() != 3) {
        std::cerr << "transformCoordinates(): ERROR: etas grid must have depth 3,  got " << etas.depth() << std::endl;
    }

    Image<float> etas_out(etas.height(), etas.width(), etas.depth());

    for (int r = 0; r < etas.height(); ++ r) {
        for (int c = 0; c < etas.width(); ++c ) {

            Eigen::Vector3f v = readVector3f(etas, r, c);
            writeVector3f(T * v, r, c, etas_out);
        }
    }

    return etas_out;
}

void retractField_orthographic(const Image<float>& etas, const Image<float>& field, Image<float>& output) {

    const int height = etas.height();
    const int width = etas.width();

#ifdef _OPENMP
    // set the number of threads reserved to Eigen to 1, so that
    // each Eigen operation in the parallel for is executed in
    // the same thread.
    int eigenThreads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel default(shared)
    {
        #pragma omp for
#endif
        for (int r = 0; r < height; r ++) {

#ifdef _OPENMP
            #pragma omp parallel shared(r, etas, field, output)
            {
                #pragma omp for
#endif
                for (int c = 0; c < width; c ++) {

                    Eigen::Vector3f eta = readVector3f(etas, r, c);
                    Eigen::Vector3f v = readVector3f(field, r, c);

                    Eigen::Vector3f mu = etaToMu_orthographic(eta, v);
                    writeVector3f(mu, r, c, output);
                }
#ifdef _OPENMP
            }
#endif
        }
#ifdef _OPENMP
    }
    // restore the number of threads allocated to Eigen
    Eigen::setNbThreads(eigenThreads);
#endif
}

Image<float> retractField_orthographic(const Image<float>&etas, const Image<float>&field) {

    Image<float> output(field.height(), field.width(), 3);
    retractField_orthographic(etas, field, output);
    return output;
}

void betapixFieldToMu(const Image<float>& etas, const Image<float>& field, Image<float>& output) {

    const int height = etas.height();
    const int width = etas.width();

#ifdef _OPENMP
    // set the number of threads reserved to Eigen to 1, so that
    // each Eigen operation in the parallel for is executed in
    // the same thread.
    int eigenThreads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel default(shared)
    {
        #pragma omp for
#endif
        for (int r = 0; r < height; r ++) {

#ifdef _OPENMP
            #pragma omp parallel shared(r, etas, field, output)
            {
                #pragma omp for
#endif
                for (int c = 0; c < width; c ++) {
                    Eigen::Vector2f beta = readVector2f(field, r, c);

                    // Orthonormal basis
                    float Bnorm = 0.0f;
                    Eigen::Matrix2Xf B = getOrthonormalBasis(etas, r, c, Bnorm);
                    Eigen::Vector3f mu = betapixToMu(B, Bnorm, beta);
                    writeVector3f(mu, r, c, output);
                }
#ifdef _OPENMP
            }
#endif
        }
#ifdef _OPENMP
    }
    // restore the number of threads allocated to Eigen
    Eigen::setNbThreads(eigenThreads);
#endif
}

Image<float> betapixFieldToMu(const Image<float>& etas, const Image<float>& field) {

    Image<float> muOutput(field.height(), field.width(), 3);
    betapixFieldToMu(etas, field, muOutput);
    return muOutput;
}

void muFieldToBetapix(const Image<float>& etas, const Image<float>& field, Image<float>& output) {

    const int height = etas.height();
    const int width = etas.width();

#ifdef _OPENMP
    // set the number of threads reserved to Eigen to 1, so that
    // each Eigen operation in the parallel for is executed in
    // the same thread.
    int eigenThreads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel default(shared)
    {
        #pragma omp for
#endif
        for (int r = 0; r < height; r ++) {

#ifdef _OPENMP
            #pragma omp parallel shared(r, etas, field, output)
            {
                #pragma omp for
#endif
                for (int c = 0; c < width; c ++) {
                    Eigen::Vector3f mu = readVector3f(field, r, c);

                    // Orthonormal basis
                    float Bnorm = 0.0f;
                    Eigen::Matrix2Xf B = getOrthonormalBasis(etas, r, c, Bnorm);

                    Eigen::Vector2f beta = muToBetapix(B, Bnorm, mu);
                    // Eigen::Vector2f beta = B*mu;
                    writeVector2f(beta, r, c, output);
                }
#ifdef _OPENMP
            }
#endif
        }
#ifdef _OPENMP
    }
    // restore the number of threads allocated to Eigen
    Eigen::setNbThreads(eigenThreads);
#endif
}

Image<float> muFieldToBetapix(const Image<float>& etas, const Image<float>& field) {

    Image<float> muOutput(field.height(), field.width(), 2);
    muFieldToBetapix(etas, field, muOutput);
    return muOutput;
}


void betapixMatrixField(const Image<float>&etas,
    Image<float>& Bpix_row0, Image<float>& Bpix_row1) {

    
    const int width = etas.width();
    const int height = etas.height();

#ifdef _OPENMP
    // set the number of threads reserved to Eigen to 1, so that
    // each Eigen operation in the parallel for is executed in
    // the same thread.
    int eigenThreads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel default(shared)
    {
        #pragma omp for
#endif
        for (int r = 0; r < height; r ++) {

#ifdef _OPENMP
            #pragma omp parallel shared(r, etas, Bpix_row0, Bpix_row1)
            {
                #pragma omp for
#endif
                for (int c = 0; c < width; c ++) {

                    // Orthonormal basis
                    float Bnorm = 0.0f;
                    Eigen::Matrix2Xf B = getOrthonormalBasis(etas, r, c, Bnorm);

                    B /= Bnorm;

                    Eigen::Vector3f row0 = B.block<1, 3>(0,0);
                    Eigen::Vector3f row1 = B.block<1, 3>(1,0);

                    writeVector3f(row0, r, c, Bpix_row0);
                    writeVector3f(row1, r, c, Bpix_row1);
                }
#ifdef _OPENMP
            }
#endif
        }
#ifdef _OPENMP
    }
    // restore the number of threads allocated to Eigen
    Eigen::setNbThreads(eigenThreads);
#endif

}


void dotProductField3(const Image<float>& field1, const Image<float>& field2, Image<float>& output) {

    const int height = field1.height();
    const int width = field1.width();

#ifdef _OPENMP
    // set the number of threads reserved to Eigen to 1, so that
    // each Eigen operation in the parallel for is executed in
    // the same thread.
    int eigenThreads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel default(shared)
    {
        #pragma omp for
#endif
        for (int r = 0; r < height; r ++) {

#ifdef _OPENMP
            #pragma omp parallel shared(r, field1, field2, output)
            {
                #pragma omp for
#endif
                for (int c = 0; c < width; c ++) {
                    Eigen::Vector3f v1 = readVector3f(field1, r, c);
                    Eigen::Vector3f v2 = readVector3f(field2, r, c);

                    output(r, c) = v1.dot(v2);
                }
#ifdef _OPENMP
            }
#endif
        }
#ifdef _OPENMP
    }
    // restore the number of threads allocated to Eigen
    Eigen::setNbThreads(eigenThreads);
#endif

}

Image<float> dotProductField3(const Image<float>& field1, const Image<float>& field2) {

    Image<float> output(field1.height(), field1.width());
    dotProductField3(field1, field2, output);
    return output;
}


void crossProductField3(const Image<float>& field1, const Image<float>& field2, Image<float>& output) {

    const int height = field1.height();
    const int width = field1.width();

#ifdef _OPENMP
    // set the number of threads reserved to Eigen to 1, so that
    // each Eigen operation in the parallel for is executed in
    // the same thread.
    int eigenThreads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel default(shared)
    {
        #pragma omp for
#endif
        for (int r = 0; r < height; r ++) {

#ifdef _OPENMP
            #pragma omp parallel shared(r, field1, field2, output)
            {
                #pragma omp for
#endif
                for (int c = 0; c < width; c ++) {
                    Eigen::Vector3f v1 = readVector3f(field1, r, c);
                    Eigen::Vector3f v2 = readVector3f(field2, r, c);

                    Eigen::Vector3f vcross = v1.cross(v2);
                    writeVector3f(vcross, r, c, output);
                    // output(r, c) = v1.dot(v2);
                }
#ifdef _OPENMP
            }
#endif
        }
#ifdef _OPENMP
    }
    // restore the number of threads allocated to Eigen
    Eigen::setNbThreads(eigenThreads);
#endif

}


Image<float> crossProductField3(const Image<float>& field1, const Image<float>& field2) {

    Image<float> output(field1.height(), field1.width(), 3);
    crossProductField3(field1, field2, output);
    return output;
}


void angleBetweenNeighbors(const Image<float>& etas, Image<float>& theta_out) {

    for(int r = 0; r < etas.height(); r ++) {
        for(int c = 0; c < etas.width(); c ++) {


            // read the base vector from the grid
            Eigen::Vector3f eta_0 = readVector3f(etas, r, c);

            // offset sign for reading eta from grid
            // if sign is negative, eta is read from row row -1, and mu needs to be mirrowed
            int sign_r = r < etas.height() - 1 ? 1 : -1;
            int sign_c = c < etas.width() - 1 ? 1 : -1;

            // read neighbors from grid
            Eigen::Vector3f eta_r = sign_r * readVector3f(etas, r + sign_r, c);
            Eigen::Vector3f eta_c = sign_c * readVector3f(etas, r, c + sign_c);

            // map to tangent space and normalize
            Eigen::Vector3f nu_r = etaToMu_orthographic(eta_0, eta_r).normalized();
            Eigen::Vector3f nu_c = etaToMu_orthographic(eta_0, eta_c).normalized();

            theta_out(r, c) = acosf(nu_r.dot(nu_c));
        }
    }
}

Image<float> angleBetweenNeighbors(const Image<float>& etas) {

    Image<float> theta_out(etas.height(), etas.width());
    theta_out.fill(0);

    angleBetweenNeighbors(etas, theta_out);

    return theta_out;
}


void distanceBetweenNeighbors(const Image<float>& etas, Image<float>& distance_out) {

    for(int r = 0; r < etas.height(); r ++) {
        for(int c = 0; c < etas.width(); c ++) {

            // get orthonormal basis (handles border)
            float Bnorm = 0.0f;
            Eigen::Matrix2Xf B = getOrthonormalBasis(etas, r, c, Bnorm);

            distance_out(r, c) = Bnorm;

            // std::cout << r << ":" << c << ": " << Bnorm << std::endl;
        }
    }
}

Image<float> distanceBetweenNeighbors(const Image<float>& etas) {

    Image<float> distance_out(etas.height(), etas.width());
    distance_out.fill(0);

    distanceBetweenNeighbors(etas, distance_out);

    return distance_out;
}

}; // namespace spherepix
