/**
 * image.h
 */

#ifndef SPHEREPIX_IMAGE_H_
#define SPHEREPIX_IMAGE_H_

#include <cstddef>
#include <cstring>
#include <cmath>
#include <memory>
#include <iostream>
#include <fstream>

#include "Eigen/Dense"

#include "spherepix/empty_delete.h"

namespace spherepix {

/**
 * \brief image type containing metadata and memory buffer pointer
 */
typedef struct {
    std::size_t height;
    std::size_t width;
    std::size_t depth;
    std::size_t pitch;       // row pitch in bytes
    std::size_t itemSize;    // element size in bytes
    void* ptr;               // image memory buffer
} image_t;


template<typename T>
class Image {

public:
    /**
     * \brief Creates an empty image
     */
    Image() {
        __height = 0;
        __width = 0;
        __depth = 0;
        __length = 0;
        __rowLength = 0;
        __pitch = 0;
    }

    /**
     * \brief Creates a new image. Storage memory owned by this object
     *
     * \param height image height
     * \param width image width
     * \param depth default(1) image depth
     */
    Image(const int height, const int width, const int depth = 1) :
        Image<T> {height, width, depth, true} {
    }

    /**
     * \brief Creates and image object with storage buffer managed externally
     *
     * \param height image height
     * \param width image width
     * \param depth default(1) image depth
     * \param data externally allocated storage space for image elements
     *
     * \warning When this object is deleted, storage buffer is not released.
     */
    Image(const int height, const int width, const int depth, T* data) :
        Image<T> {height, width, depth, false, data} {
    }

    ~Image() {
        // nothing to do
    }

    /**
     * \return buffer pointer
     */
    inline T* data() {return __data.get();}

    /**
     * \return buffer pointer
     */
    inline const T* dataConst() const {return __data.get();}

    /**
     * \return image height
     */
    inline int height() const  {return __height;}

    /**
     * \return image width
     */
    inline int width() const   {return __width;}

    /**
     * \return image depth
     */
    inline int depth() const   {return __depth;}

    /**
     * \return the number of elements per row
     */
    inline int rowLength() const {return __rowLength;}

    /**
     * \return pitch in bytes
     */
    inline int pitch() const   {return __pitch;}

    /**
     * \return total elements in the image
     */
    inline int length() const  {return __length;}

    /**
     * \return size in bytes of allocated space.
     */
    inline int sizeBytes() const {return __length * sizeof(T);}

    inline bool compareShape(const Image<T>& other) const {
        return __height == other.__height &&
               __width == other.__width &&
               __depth == other.__depth;
    }

    /**
     * \brief test if the provided coordinates are within the image boundaries
     *
     * \param row coordinate
     * \param col coordinate
     *
     * \return true if row in [0, height -1] and col in [0, width -1]
     */
    inline bool checkCoordinates(const int row, const int col) const {
        return row >= 0 && row < __height &&
               col >= 0 && col < __width;
    }

    /**
     * \brief test if the provided coordinates are within the image boundaries
     *
     * \param row coordinate
     * \param col coordinate
     * \param channel coordinates
     *
     * \return true if row in [0, height -1] and col in [0, width -1] and channel in [0, depth -1]
     */
    inline bool checkCoordinates(const int row, const int col, const int channel) const {
        return row >= 0 && row < __height &&
               col >= 0 && col < __width &&
               channel >= 0 && channel < __depth;
    }

    Image<T> copy() const {

        Image<T> imgCopy(__height, __width, __depth);
        memcpy((void*)&imgCopy[0], (void*)&(*this)[0], sizeBytes());
        return imgCopy;
    }

    void copyFrom(const Image<T>& img) {

        if (img.__height == __height &&
                img.__width == __width &&
                img.__depth == __depth) {

            memcpy((void*)&(*this)[0], (void*)img.dataConst(), sizeBytes());
        } else {
            std::cerr << "Image::copyFrom(): ERROR: image shapes do not match" << std::endl;
        }
    }

    /**
     * \brief returns a new image with the elements taken from a region of this image
     *
     * \param row initial row coordinate
     * \param col initial column coordinate
     * \param height subimage height
     * \param width subimage width
     */
    Image<T> subImage(const int row, const int col, const int height, const int width) const {

        Image<T> subImg(height, width, __depth);
        for (int r = 0; r < height; ++ r) {
            for (int c = 0; c < width; ++ c) {
                for (int d = 0; d < __depth; ++ d) {
                    subImg(r, c, d) = (*this)(r + row, c + col, d);
                }
            }
        }
        return subImg;
    }

    void save(const std::string& fileName) {

        std::ofstream fileWriter(fileName, std::ios_base::binary | std::ios::ate);
        fileWriter.write((const char*)&(*this)[0], sizeBytes());
        fileWriter.close();
    }

    void fill(const T& value) {
        // std::cout << "Image::setTo(): value: " << value << std::endl;
        for (int i = 0; i < __length; ++ i) {
            (*this)[i] = value;
        }
    }

    void clear() {
        memset((void*)__data.get(), 0, sizeBytes());
    }

    inline T& operator[](std::size_t idx) {
        return __data.get()[idx];
    }

    inline const T& operator[](std::size_t idx) const {
        return __data.get()[idx];
    }

    inline T& operator()(const int row, const int col, const int channel = 0) {
        return __data.get()[(row * __rowLength) + (col * __depth) + channel];
    }

    inline const T& operator()(const int row, const int col, const int channel = 0) const {
        return __data.get()[(row * __rowLength) + (col * __depth) + channel];
    }

    inline void get(const int row, const int col, std::vector<T>& out) const {
        const int offset = row*__rowLength + col*__depth;
        for(int d = 0; d < __depth; d++) {
            out[d] = (*this)[offset + d];
        }
    }

    inline void set(const int row, const int col, const std::vector<T>& value) {
        const int offset = row*__rowLength + col*__depth;
        for(int d = 0; d < __depth; d++) {
            (*this)[offset + d] = value[d];
        }
    }

    image_t asImage_t() {
        
        image_t desc;
        
        desc.height = __height;
        desc.width = __width;
        desc.depth = __depth;
        desc.pitch = __pitch;
        desc.itemSize = sizeof(T);
        desc.ptr = (void*)data();
        return desc;
    }


private:

    Image(const int height, const int width, const int depth,
          const bool allocate, T* data = 0) {

        //FIXME: validate image size, throw exception
        __height = height;
        __width = width;
        __depth = depth;

        __rowLength = __width * __depth;
        __length = __height * __rowLength;

        // pitch in bytes
        __pitch = __rowLength * sizeof(T);

        // if the object allocates its own data storage space
        if (allocate) {
            __data = std::shared_ptr<T> {
                new T[__length],
                std::default_delete<T[]>()
            };
        } else {

            // creates a shared pointer with an empty_delete deleter
            // this way, data buffer is not deleted by this object but
            // by its original owner
            __data = std::shared_ptr<T> { data, array_empty_deleter<T>() };
        }
    }

    std::size_t __height;
    std::size_t __width;
    std::size_t __depth;
    std::size_t __length;
    std::size_t __rowLength;
    std::size_t __pitch;
    std::shared_ptr<T> __data;
};


//#########################################################
// UTILITY METHODS
//#########################################################

/**
 * \brief reads a Vector3f from an image
 */
template<bool checkCoordinates = true>
Eigen::Vector3f readVector3f(const Image<float>& image, const int row, const int col);

/**
 * \brief writes a Vector3f to an image
 */
template<bool checkCoordinates = true>
void writeVector3f(const Eigen::Vector3f& v, const int row, const int col, Image<float>& image);

/**
 * \brief reads a Vector2f from an image
 */
template<bool checkCoordinates = true>
Eigen::Vector2f readVector2f(const Image<float>& image, const int row, const int col);

/**
 * \brief writes a Vector2f to an image
 */
template<bool checkCoordinates = true>
void writeVector2f(const Eigen::Vector2f& v, const int row, const int col, Image<float>& image);


//#########################################################
// INTERPOLATION METHODS
//#########################################################

template<typename T>
void interpolate(const Image<T>& img, const float row, const float col, std::vector<T>& out);

template<typename T>
std::vector<T> interpolate(const Image<T>& img, const float row, const float col);

template<typename T>
T interpolate(const Image<T>& img, const float row, const float col);

template<typename T>
void interpolateImage(const Image<T>& inputImage, const Image<float>& coordinates,
                      Image<T>& outputImage);


//#########################################################
// CONVOLUTION METHODS
//#########################################################

// TODO: need to add border policy

template<typename T>
Image<T> convolve2D(const Image<T>& img, const Image<T>& mask);

template<typename T>
void convolve2D(const Image<T>& img, const Image<T>& mask, Image<T>& output);

template<typename T>
Image<T> convolveRow(const Image<T>& img, const Image<T>& mask);

template<typename T>
void convolveRow(const Image<T>& img, const Image<T>& mask, Image<T>& output);

template<typename T>
Image<T> convolveColumn(const Image<T>& img, const Image<T>& mask);

template<typename T>
void convolveColumn(const Image<T>& img, const Image<T>& mask, Image<T>& output);

//#########################################################
// IMPLEMENTATION OF TEMPLATE METHODS
//#########################################################
#include "spherepix/image_impl.h"

}; // namespace spherepix


#endif // SPHEREPIX_IMAGE_H_