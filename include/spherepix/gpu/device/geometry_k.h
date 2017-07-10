/**
 * \file geometry_k.h
 * \brief Device functions for sphere geometry.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef SPHEREPIX_GPU_GEOMETRY_K_H_
#define SPHEREPIX_GPU_GEOMETRY_K_H_

// #include "spherepix/gpu/math.h"
#include "spherepix/gpu/device/math_k.h"
#include "spherepix/gpu/device/pixelation_k.h"

namespace spherepix {
namespace gpu {


inline __device__ float3 etaToMu_orthographic(const float3& eta_0, const float3& eta) {

    // elements of matrix (eta_0 * eta_0^T)
    const float xx = eta_0.x * eta_0.x;
    const float xy = eta_0.x * eta_0.y;
    const float xz = eta_0.x * eta_0.z;
    const float yy = eta_0.y * eta_0.y;
    const float yz = eta_0.y * eta_0.z;
    const float zz = eta_0.z * eta_0.z;

    // mu = (I - (eta_0 * eta_0^T)) eta
    float3 mu = make_float3(
        (1 - xx)*eta.x -       xy*eta.y -       xz*eta.z,
             -xy*eta.x + (1 - yy)*eta.y -       yz*eta.z,
             -xz*eta.x -       yz*eta.y + (1 - zz)*eta.z
        );

    return mu;
}

/**
 * \brief Returns unnormalized B matrix.
 */
inline __device__ matrix23F_t betaMatrix(const pixface_t& face, const float3& eta_0, const int2 pix) {

    int sign = pix.x < face.width -1? 1 : -1;
    float3 eta_col = pixfaceEta(face, make_int2(pix.x + 1, pix.y));

    float3 mu_0 = sign*etaToMu_orthographic(eta_0, eta_col);

    matrix23F_t B;
    B.row0 = sign*mu_0;
    // B.row1 = -1.0f*cross(eta_0, mu_0);
    B.row1 = cross(mu_0, eta_0);
    return B;
}

inline __device__ matrix23F_t orthonormalBasis(const pixface_t& face, const float3& eta_0, const int2 pix) {

    matrix23F_t B = betaMatrix(face, eta_0, pix);
    
    normalize(B.row0);
    normalize(B.row1);
    return B;
}

inline __device__ matrix23F_t orthonormalBasis(const pixface_t& face, const float3& eta_0,
    const int2 pix, float& Bnorm) {

    matrix23F_t B = betaMatrix(face, eta_0, pix);

    Bnorm = norm(B.row0);
    B.row0 *= (1.0/Bnorm);
    
    normalize(B.row1);
    return B;
}

inline __device__ matrix23F_t orthonormalBasis(const pixface_t& face, const int2 pix) {

    return orthonormalBasis(face, pixfaceEta(face, pix), pix);
}

inline __device__ matrix23F_t betaPixMatrix(const pixface_t& face, const float3& eta_0, const int2 pix) {

    matrix23F_t B = betaMatrix(face, eta_0, pix);
    B *= invnormsqr(B.row0);
    return B;
}

inline __device__ matrix23F_t betaPixMatrix(const pixface_t& face, const int2 pix) {
    return betaPixMatrix(face, pixfaceEta(face, pix), pix);
}


inline __device__ float3 betapixToMu(const pixface_t& face,
    const float3& eta_0, const int2& pix,
    const float2& beta) {

    matrix23F_t B = betaMatrix(face, eta_0, pix);
    return transposeMul(B, beta);
}

inline __device__ float3 betapixToMu(const pixface_t& face,
    const int2& pix, const float2& beta) {
    return betapixToMu(face, pixfaceEta(face, pix), pix, beta);
}

inline __device__ float3 betapixToMu(const matrix23F_t& B, const float Bnorm,
    const float2& beta) {

    return Bnorm*transposeMul(B, beta);
}

} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_GEOMETRY_K_H_