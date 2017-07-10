/**
 * \file math_k.h
 * \brief Device math functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef SPHEREPIX_GPU_MATH_K_H_
#define SPHEREPIX_GPU_MATH_K_H_

#include "spherepix/gpu/math.h"

namespace spherepix {
namespace gpu {


//###############################################
// INT2 OPERATIONS
//###############################################

inline __device__ bool operator ==(const int2& a, const int2& b) {
    return a.x == b.x && a.y == b.y;
}

inline __device__ bool operator !=(const int2& a, const int2& b) {
    return a.x != b.x || a.y != b.y;
}

inline __device__ int2 operator +(const int2& a, const int2& b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator *(const float a, const int2&b) {
    return make_float2(a*b.x, a*b.y);
}

//###############################################
// FLOAT2 OPERATIONS
//###############################################
inline __device__ float2 swap(const float2& a) {
    float aux = a.x;
    return make_float2(a.y, aux);
}

inline __device__ bool inRange(const float2& a, const float2& b) {
    return a.x >= 0 && a.x < b.x && a.y >= 0 && a.y < b.y;
}


//###############################################
// FLOAT3 OPERATIONS
//###############################################

//###########################
// CONSTRUCTORS
//###########################

inline __device__ float3 _make_float3(const float4& a) {
    return make_float3(a.x, a.y, a.z);
}

//###########################
// OPERATORS
//###########################

inline __device__ void operator *=(float3&v , const float a) {
    v.x *= a; v.y *= a; v.z *= a;
}

/** float3 c = a + b; */
inline __device__ float3 operator +(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/** a += b*/
inline __device__ void operator +=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

/** float3 c = a - b; */
inline __device__ float3 operator -(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator *(const int a, const float3& v) {
    return make_float3(a*v.x, a*v.y, a*v.z);
}

inline __device__ float3 operator *(const float a, const float3& v) {
    return make_float3(a*v.x, a*v.y, a*v.z);
}

inline __device__ float3 operator *(const float3& v, const float a) {
    return make_float3(a*v.x, a*v.y, a*v.z);
}


//###########################
// FUNCTIONS
//###########################

inline __device__ void print(const float3& v) {
    printf("[%f, %f, %f]\n", v.x, v.y, v.z);
}

inline __device__ float dot(const float3& v0, const float3& v1) {
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

inline __device__ float3 cross(const float3& a, const float3& b) {

    return make_float3( a.y*b.z - a.z*b.y,
                        a.z*b.x - a.x*b.z,
                        a.x*b.y - a.y*b.x);
}

inline __device__ float norm(const float3& v) {
    return sqrtf(dot(v,v));
}

inline __device__ float invnorm(const float3& v) {
    return 1.0/sqrtf(dot(v,v));
}

inline __device__ float invnormsqr(const float3& v) {
    return 1.0/dot(v,v);
}

inline __device__ void normalize(float3& v) {
    v *= invnorm(v);
}

inline __device__ matrix33F_t sqrmatrix(const float3& v) {

    // elements of matrix (v * v^T)
    const float xx = v.x * v.x;
    const float xy = v.x * v.y;
    const float xz = v.x * v.z;
    const float yy = v.y * v.y;
    const float yz = v.y * v.z;
    const float zz = v.z * v.z;

    matrix33F_t S;
    S.row0 = make_float3(xx, xy, xz);
    S.row1 = make_float3(xy, yy, yz);
    S.row2 = make_float3(xz, yz, zz);
    return S;
}

//###############################################
// FLOAT4 OPERATIONS
//###############################################

inline __device__ void print(const float4& v) {
    printf("[%e, %e, %e, %e]\n", v.x, v.y, v.z, v.w);
}

inline __device__ float4 _make_float4(const float3& a, const float w = 0.0) {
    return make_float4(a.x, a.y, a.z, w);
}

/** float4 c = a + b; */
inline __device__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/** a += b */
inline __device__ void operator +=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

/** float4 c = a - b; */
inline __device__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/** float4 a; float b; float4 c = a * b */
inline __device__ float4 operator*(const float4& a, const float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/** float4 a; float b; float4 c = b * a */
inline __device__ float4 operator*(const float b, const float4& a) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/** float4 a; float4 c; c -= a; */
inline __device__ void operator-=(float4& a, const float4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}


//###############################################
// MAT33F OPERATIONS
//###############################################

//###########################
// CONSTRUCTORS
//###########################

inline __device__ matrix33F_t make_matrix33F() {
    matrix33F_t M;
    M.row0 = make_float3(0, 0, 0);
    M.row1 = make_float3(0, 0, 0);
    M.row2 = make_float3(0, 0, 0);
    return M;
}


//###########################
// OPERATORS
//###########################

/** M += A */
inline __device__ void operator +=(matrix33F_t& M, const matrix33F_t& A) {
    M.row0 += A.row0;
    M.row1 += A.row1;
    M.row2 += A.row2;
}


inline __device__ matrix33F_t operator *(const float a, const matrix33F_t& A) {
    matrix33F_t M;
    M.row0 = a*A.row0;
    M.row1 = a*A.row1;
    M.row2 = a*A.row2;
    return M;
}

inline __device__ matrix33F_t operator *(const matrix33F_t& A, const float a) {
    matrix33F_t M;
    M.row0 = a*A.row0;
    M.row1 = a*A.row1;
    M.row2 = a*A.row2;
    return M;
}


//###########################
// FUNCTIONS
//###########################

inline __device__ matrix33F_t eye33() {
    matrix33F_t M;
    M.row0 = make_float3(1, 0, 0);
    M.row1 = make_float3(0, 1, 0);
    M.row2 = make_float3(0, 0, 1);
    return M;   
}


//###############################################
// MAT44F OPERATIONS
//###############################################

//###########################
// CONSTRUCTORS
//###########################

inline __device__ matrix44F_t make_matrix44F() {
    matrix44F_t M;
    M.row0 = make_float4(0, 0, 0, 0);
    M.row1 = make_float4(0, 0, 0, 0);
    M.row2 = make_float4(0, 0, 0, 0);
    M.row3 = make_float4(0, 0, 0, 0);
    return M;
}

inline __device__ matrix44F_t make_matrix44F(const matrix33F_t& A, const float fill=0) {
    matrix44F_t M;
    M.row0 = _make_float4(A.row0, fill);
    M.row1 = _make_float4(A.row1, fill);
    M.row2 = _make_float4(A.row2, fill);
    M.row3 = make_float4(fill, fill, fill, fill);
    return M;
}


//###########################
// OPERATORS
//###########################

inline __device__ void print(const matrix44F_t& A) {
    print(A.row0);
    print(A.row1);
    print(A.row2);
    print(A.row3);
}

inline __device__ matrix44F_t operator +(const matrix44F_t& A, const matrix44F_t& B) {
    matrix44F_t M;
    M.row0 = A.row0 + B.row0;
    M.row1 = A.row1 + B.row1;
    M.row2 = A.row2 + B.row2;
    M.row3 = A.row3 + B.row3;
    return M;
}

inline __device__ matrix44F_t operator *(const float a, const matrix44F_t& A) {
    matrix44F_t M;
    M.row0 = a*A.row0;
    M.row1 = a*A.row1;
    M.row2 = a*A.row2;
    M.row3 = a*A.row3;
    return M;
}

inline __device__ matrix44F_t operator *(const matrix44F_t& A, const float a) {
    matrix44F_t M;
    M.row0 = a*A.row0;
    M.row1 = a*A.row1;
    M.row2 = a*A.row2;
    M.row3 = a*A.row3;
    return M;
}

//###########################
// FUNCTIONS
//###########################

inline __device__ matrix44F_t eye44() {
    matrix44F_t M;
    M.row0 = make_float4(1, 0, 0, 0);
    M.row1 = make_float4(0, 1, 0, 0);
    M.row2 = make_float4(0, 0, 1, 0);
    M.row3 = make_float4(0, 0, 0, 1);
    return M;   
}

inline __device__ matrix44F_t diag44(const float d) {
    matrix44F_t M;
    M.row0 = make_float4(d, 0, 0, 0);
    M.row1 = make_float4(0, d, 0, 0);
    M.row2 = make_float4(0, 0, d, 0);
    M.row3 = make_float4(0, 0, 0, d);
    return M;      
}

/**
 * \brief Returns the Cholesky decomposition of A
 *
 * A = LL^T
 *
 * The method uses the Cholesky-Banachiewicz algorithm
 * (row by row) to compute L. See wikipedia.
 */
inline __device__ matrix44F_t cholesky(const matrix44F_t& A) {

    matrix44F_t L = make_matrix44F();

    // holds 1 / diag(L)
    float3 invdiag;

    // first row
    L.row0.x = sqrtf(A.row0.x);
    invdiag.x = 1.0 / L.row0.x;
    
    // second row
    L.row1.x = (invdiag.x)*(A.row1.x);
    L.row1.y = sqrtf(A.row1.y - L.row1.x*L.row1.x);
    invdiag.y = 1.0 / L.row1.y;
    
    // third row
    L.row2.x = (invdiag.x)*(A.row2.x);
    L.row2.y = (invdiag.y)*(A.row2.y - (L.row2.x*L.row1.x));
    L.row2.z = sqrtf(A.row2.z - (L.row2.x*L.row2.x + L.row2.y*L.row2.y));
    invdiag.z = 1.0 / L.row2.z;
    
    // fourth row
    L.row3.x = (invdiag.x)*(A.row3.x);
    L.row3.y = (invdiag.y)*(A.row3.y - (L.row3.x*L.row1.x));
    L.row3.z = (invdiag.z)*(A.row3.z - (L.row3.x*L.row2.x + L.row3.y*L.row2.y));
    L.row3.w = sqrtf(A.row3.w - (L.row3.x*L.row3.x + L.row3.y*L.row3.y + L.row3.z*L.row3.z));

    return L;
}


inline __device__ float4 forwardSubstitution(const matrix44F_t& A, const float4& b) {

    float4 x;

    x.x = b.x / A.row0.x;
    x.y = (b.y - A.row1.x*x.x) / A.row1.y;
    x.z = (b.z - A.row2.x*x.x - A.row2.y*x.y) / A.row2.z;
    x.w = (b.w - A.row3.x*x.x - A.row3.y*x.y - A.row3.z*x.z) / A.row3.w;

    return x;
}


inline __device__ float4 backwardSubstitution(const matrix44F_t& A, const float4& b) {

    float4 x;

    // solves the triangular backward substitution 
    // assuming A is entered as lower triangular.

    x.w = b.w / A.row3.w;
    x.z = (b.z - A.row3.z*x.w) / A.row2.z;
    x.y = (b.y - A.row3.y*x.w - A.row2.y*x.z) / A.row1.y;
    x.x = (b.x - A.row3.x*x.w - A.row2.x*x.z - A.row1.x*x.y) / A.row0.x;

    return x;
}

inline __device__ float4 solveCholesky(const matrix44F_t& A, const float4& b) {
    
    // cholesky decomposition
    matrix44F_t L = cholesky(A);

    // solve Ly = b
    float4 y = forwardSubstitution(L, b);

    // then solve L.T x = y
    // where L.T is upper triangular
    float4 x = backwardSubstitution(L, y);

    return x;
}

//###############################################
// MAT23F OPERATIONS
//###############################################

inline __device__ matrix23F_t make_matrix23F() {
    matrix23F_t M;
    M.row0 = make_float3(0, 0, 0);
    M.row1 = make_float3(0, 0, 0);
    return M;
}

/**
 * matrix23F_t A; float3 x;
 * float2 y = A * b
 */
inline __device__ float2 operator*(const matrix23F_t& A, const float3& x) {
    return make_float2(dot(A.row0, x), dot(A.row1, x));
}

inline __device__ void operator*=(matrix23F_t& A, const float a) {
    A.row0 *= a;
    A.row1 *= a;
}

/**
 * float3 v = A.T * x
 */
inline __device__ float3 transposeMul(const matrix23F_t& A, const float2& x) {

    return A.row0*x.x + A.row1*x.y;
    // return make_float3( A.row0.x*x.x + A.row1.x*x.y,
    //                     A.row0.y*x.x + A.row1.y*x.y,
    //                     A.row0.z*x.x + A.row1.z*x.y);
}

} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_MATH_K_H_