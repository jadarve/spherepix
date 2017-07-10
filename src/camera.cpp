
#include <exception>
#include <iostream>

#include "spherepix/camera.h"


using namespace std;


namespace spherepix {

//#########################################################
// Camera
//#########################################################
Camera::~Camera() {
    // nothing to do
}


//#########################################################
// PinholeCamera
//#########################################################

PinholeCamera::PinholeCamera(const float focalLength, const int height, const int width,
                             const float sensorHeight, const float sensorWidth) {

    __height = height;
    __width = width;

    int imgX = __width - 1;
    int imgY = __height -1;
    __K << focalLength*imgX / sensorWidth, 0, imgX*0.5f,
            0, focalLength*imgY/sensorHeight, imgY*0.5f,
            0, 0, 1;

    __Kinv = __K.inverse();
    __computeCoordinates();
}

PinholeCamera::PinholeCamera(const int height, const int width,
        Eigen::Matrix3f& intrinsics) {

    __height = height;
    __width = width;

    __K = intrinsics;
    __Kinv = __K.inverse();
    __computeCoordinates();
}

PinholeCamera::~PinholeCamera() {
    // nothing to do
}

const Image<float> PinholeCamera::sphericalCoordinates() {
    return __sphericalCoordinates;
}

const Image<float> PinholeCamera::surfaceCoordinates() {
    return __planeCoordinates;
}

int PinholeCamera::height() const {
    return __height;
}

int PinholeCamera::width() const {
    return __width;
}

bool PinholeCamera::isVerticalFlipped() const {
    return true;
}

Eigen::Matrix3f PinholeCamera::intrinsicsMatrix() const {
    return __K;
}

void PinholeCamera::__computeCoordinates() {

    __sphericalCoordinates = Image<float>(__height, __width, 3);
    __planeCoordinates = Image<float>(__height, __width, 3);

    for(int r = 0; r < __height; ++ r) {
        for(int c = 0; c < __width; ++ c) {

            // flip the column and row coordinates, as
            // the intrinsics matrix is in (x, y, z) order

            // OLD
            // Eigen::Vector3f pixel(c, r, 1);

            // Vertically flipped row coordinate
            Eigen::Vector3f pixel(c, __height -1 -r, 1);


            Eigen::Vector3f planeCoord = __Kinv*pixel;
            Eigen::Vector3f sphereCoord = planeCoord.normalized();

            writeVector3f(planeCoord, r, c, __planeCoordinates);
            writeVector3f(sphereCoord, r, c, __sphericalCoordinates);
        }
    }
}


//#########################################################
// OmnidirectionalCamera
//#########################################################

OmnidirectionalCamera::OmnidirectionalCamera(const Image<float>& sphereCoordinates,
    const bool isVerticalFlipped) {

    __sphericalCoordinates = sphereCoordinates.copy();
    __verticalFlipped = isVerticalFlipped;
}

OmnidirectionalCamera::~OmnidirectionalCamera() {
    // nothing to do
}

const Image<float> OmnidirectionalCamera::sphericalCoordinates() {
    return __sphericalCoordinates;
}

const Image<float> OmnidirectionalCamera::surfaceCoordinates() {
    return __sphericalCoordinates;
}

int OmnidirectionalCamera::height() const {
    return __sphericalCoordinates.height();
}

int OmnidirectionalCamera::width() const {
    return __sphericalCoordinates.width();
}

bool OmnidirectionalCamera::isVerticalFlipped() const {
    return __verticalFlipped;
}

}; // namespace spherepix