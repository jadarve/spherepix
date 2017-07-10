/*
 * geometry.h
 *
 */

#ifndef SPHEREPIX_GEOMETRY_H_
#define SPHEREPIX_GEOMETRY_H_

// #include <tuple>

#include "Eigen/Dense"

#include "image.h"

using namespace Eigen;

namespace spherepix {

/////////////////////////////////////////////////
// TANGENT SPACE PROJECTION MATRIX
/////////////////////////////////////////////////

/**
 * \brief Returns the tangent space projection matrix for eta.
 *
 * \param eta reference spherical coordinate. (x, y, z) normalized
 */
Eigen::Matrix3f projectionMatrix(const Eigen::Vector3f& eta);


/////////////////////////////////////////////////
// RETRACTIONS
/////////////////////////////////////////////////

/**
 * \brief Returns the orthographic retraction of eta on tangent the space of eta_0.
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param eta point to project. (x, y, z) normalized
 *
 * \return mu coordinate (x, y, z)
 */
Eigen::Vector3f etaToMu_orthographic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta);

/**
 * \brief Returns the perspective retraction of eta on tangent the space of eta_0.
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param eta point to project. (x, y, z) normalized
 *
 * \return mu coordinate (x, y, z)
 */
Eigen::Vector3f etaToMu_perspective(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta);

/**
 * \brief Returns the geodesic retraction of eta on tangent the space of eta_0.
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param eta point to project. (x, y, z) normalized
 *
 * \return mu coordinate (x, y, z)
 */
Eigen::Vector3f etaToMu_geodesic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta);

/**
 * \brief Returns the cordal retraction of eta on tangent the space of eta_0.
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param eta point to project. (x, y, z) normalized
 *
 * \return mu coordinate (x, y, z)
 */
Eigen::Vector3f etaToMu_cordal(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta);


/////////////////////////////////////////////////
// INVERSE RETRACTIONS
/////////////////////////////////////////////////

/**
 * \brief Returns the orthographic inverse retraction of mu given reference eta_0
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param mu tangent space point. (x, y, z)
 *
 * \return eta coordinate. (x, y, z) normalized
 */
Eigen::Vector3f muToEta_orthographic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu);

/**
 * \brief Returns the perspective inverse retraction of mu given reference eta_0
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param mu tangent space point. (x, y, z)
 *
 * \return eta coordinate. (x, y, z) normalized
 */
Eigen::Vector3f muToEta_perspective(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu);

/**
 * \brief Returns the geodesic inverse retraction of mu given reference eta_0
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param mu tangent space point. (x, y, z)
 *
 * \return eta coordinate. (x, y, z) normalized
 */
Eigen::Vector3f muToEta_geodesic(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu);

/**
 * \brief Returns the cordal inverse retraction of mu given reference eta_0
 *
 * \param eta_0 reference spherical coordinate. (x, y, z) normalized
 * \param mu tangent space point. (x, y, z)
 *
 * \return eta coordinate. (x, y, z) normalized
 */
Eigen::Vector3f muToEta_cordal(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& mu);



/////////////////////////////////////////////////
// ROTATION MATRICES
/////////////////////////////////////////////////

/**
 * \brief Returns the rotation matrix between vectors eta_0 and eta_1.
 *
 * \param eta_0 first vector, (x, y, z) normalized.
 * \param eta_1 second vector, (x, y, z) normalized.
 *
 * \return 3x3 rotation matrix.
 */
Eigen::Matrix3f rotationMatrix(const Eigen::Vector3f& eta_0, const Eigen::Vector3f& eta_1);

/**
 * \brief Returns the rotation matrix given the rotation axis and angle.
 *
 * \param axis rotation axis, (x, y, z) normalized.
 * \param theta rotation angle in radians.
 *
 * \returns 3x3 rotation matrix.
 */
Eigen::Matrix3f rotationMatrixAxisAngle(const Eigen::Vector3f& axis, const float& theta);


/////////////////////////////////////////////////
// ORTHONORMAL COORDINATES
/////////////////////////////////////////////////

Eigen::Matrix2Xf getOrthonormalBasis(const Image<float>& etas, const int row, const int col);

Eigen::Matrix2Xf getOrthonormalBasis(const Image<float>& etas, const int row, const int col, float& norm_out);

Eigen::Vector2f muToBetapix(const Eigen::Matrix2Xf& B, const float Bnorm, const Eigen::Vector3f& mu);
Eigen::Vector3f betapixToMu(const Eigen::Matrix2Xf& B, const float Bnorm, const Eigen::Vector2f& beta);

void betapixFieldToMu(const Image<float>& etas, const Image<float>& field, Image<float>& output);
Image<float> betapixFieldToMu(const Image<float>& etas, const Image<float>& field);

void muFieldToBetapix(const Image<float>& etas, const Image<float>& field, Image<float>& output);
Image<float> muFieldToBetapix(const Image<float>& etas, const Image<float>& field);

void betapixMatrixField(const Image<float>&etas, Image<float>& Bpix_row0, Image<float>& Bpix_row1);

/////////////////////////////////////////////////
// COORDINATES INTERPOLATION
/////////////////////////////////////////////////

Eigen::Vector2f findInterpolationCoodinates(const Eigen::Vector3f& eta,
    const Image<float>& etaGrid, const bool flipVertical=false);

/////////////////////////////////////////////////
// UTILITY FUNCTIONS
/////////////////////////////////////////////////

void transformCoordinates(const Image<float>& etas, const Eigen::Matrix3f& T, Image<float>& etas_out);
Image<float> transformCoordinates(const Image<float>& etas, const Eigen::Matrix3f& T);

void retractField_orthographic(const Image<float>& etas, const Image<float>& field, Image<float>& output);
Image<float> retractField_orthographic(const Image<float>&etas, const Image<float>&field);


void dotProductField3(const Image<float>& field1, const Image<float>& field2, Image<float>& output);
Image<float> dotProductField3(const Image<float>& field1, const Image<float>& field2);

void crossProductField3(const Image<float>& field1, const Image<float>& field2, Image<float>& output);
Image<float> crossProductField3(const Image<float>& field1, const Image<float>& field2);

void angleBetweenNeighbors(const Image<float>& etas, Image<float>& theta_out);
Image<float> angleBetweenNeighbors(const Image<float>& etas);

void distanceBetweenNeighbors(const Image<float>& etas, Image<float>& distance_out);
Image<float> distanceBetweenNeighbors(const Image<float>& etas);

}; // namespace spherepix

#endif // SPHEREPIX_GEOMETRY_H_
