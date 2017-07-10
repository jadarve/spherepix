/*
 * camera.h
 *
 */

#ifndef SPHEREPIX_CAMERA_H_
#define SPHEREPIX_CAMERA_H_

#include "Eigen/Dense"

#include "spherepix/image.h"

namespace spherepix {

class Camera {

public:
    virtual ~Camera();

    virtual const Image<float> sphericalCoordinates() = 0;
    virtual const Image<float> surfaceCoordinates() = 0;

    virtual int height() const = 0;
    virtual int width() const = 0;

    /**
     * \brief returns true if pixel (0,0) represents top-left corner
     *  of the image, or false if it represents bottom-left.
     */
    virtual bool isVerticalFlipped() const = 0;

};


class PinholeCamera : public Camera {

public:
    PinholeCamera(const float focalLength, const int height, const int width,
        const float sensorHeight, const float sensorWidth);

    PinholeCamera(const int height, const int width,
        Eigen::Matrix3f& intrinsics);

    ~PinholeCamera();

    const Image<float> sphericalCoordinates();
    const Image<float> surfaceCoordinates();

    Eigen::Matrix3f intrinsicsMatrix() const;

    int height() const;
    int width() const;
    bool isVerticalFlipped() const;


private:
    void __computeCoordinates();

private:
    // float __focalLength;
    // float __sensorHeight;
    // float __sensorWidth;
    int __height;
    int __width;

    // intrinsics matrix
    Eigen::Matrix3f __K;
    Eigen::Matrix3f __Kinv;

    // Spherical coordinates
    Image<float> __sphericalCoordinates;
    Image<float> __planeCoordinates;
};


class OmnidirectionalCamera : public Camera {

public:
    OmnidirectionalCamera(const Image<float>& sphereCoordinates,
        const bool isVerticalFlipped=false);
    ~OmnidirectionalCamera();

    const Image<float> sphericalCoordinates();
    const Image<float> surfaceCoordinates();

    int height() const;
    int width() const;
    bool isVerticalFlipped() const;


private:
    Image<float> __sphericalCoordinates;
    bool __verticalFlipped;
};


}; // namespace spherepix

#endif // SPHEREPIX_CAMERA_H_
