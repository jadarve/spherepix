/**
 * \file pixelation_k.h
 * \brief Device functions to read pixelation data.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#ifndef SPHEREPIX_GPU_PIXELATION_K_H_
#define SPHEREPIX_GPU_PIXELATION_K_H_

#include <stdio.h>

#include <flowfilter/gpu/device/image_k.h>

#include "spherepix/gpu/pixelation.h"
#include "spherepix/gpu/math.h"

#include "spherepix/gpu/device/math_k.h"
// #include "spherepix/gpu/device/geometry_k.h"

using namespace flowfilter::gpu;

namespace spherepix {
namespace gpu {


template<typename T>
inline __device__ T pixfaceimgRead(const pixfaceimg_t<T>& img, const int2& pix) {
    return tex2D<T>(img.imgTexture, pix.x, pix.y);
}


template<typename T>
inline __device__ T pixfaceimgRead(const pixfaceimg_t<T>& img, const float2& pix) {
    return tex2D<T>(img.imgTexture, pix.x, pix.y);
}


template<typename T>
inline __device__ void pixfaceimgWrite(const pixfaceimg_t<T>& img, const int2& pix, const T& v) {
    *coordPitch(img.img, pix) = v;
}


inline __device__ matrix23F_t pixfaceimgReadMat23F(const pixfaceimgMat23F_t& img,
    const int2 pix) {

    matrix23F_t M;
    M.row0 = _make_float3(pixfaceimgRead(img.row0, pix));
    M.row1 = _make_float3(pixfaceimgRead(img.row1, pix));
    return M;
}


inline __device__ float3 pixfaceEta(const pixface_t& face, const int2 pix) {

    // TODO: border conditions?
    // return _make_float3(tex2D<float4>(face.etasTexture, pix.x, pix.y));
    return _make_float3(pixfaceimgRead(face.etas, pix));
}


// inline __device__ matrix23F_t pixfaceBpix(const pixface_t& face, const int2 pix) {
    
//     // TODO: border conditions?
//     return pixfaceimgReadMat23F(face.Bpix, pix);
// }


// inline __device__ matrix23F_t pixfaceBpix(const pixface_t& face, const int2 pix) {
    
//     float3 eta_0 = pixfaceEta(face, pix);

//     int sign = pix.y < face.height -1? 1 : -1;
//     float3 eta_row = pixfaceEta(face, make_int2(pix.x, pix.y + sign));

//     float3 mu_0 = sign*etaToMu_orthographic(eta_0, eta_row);

//     // inverse norm square
//     float invNorm = invnormsqr(mu_0);

//     matrix23F_t Bpix;
//     Bpix.row0 = (invNorm*sign)*mu_0;
//     Bpix.row1 = invNorm*cross(mu_0, eta_0);
//     return Bpix;
// }


} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_PIXELATION_K_H_