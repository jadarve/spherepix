/*
 * pixelation.h
 */

#ifndef SPHEREPIX_PIXELATION_H_
#define SPHEREPIX_PIXELATION_H_

#include <cmath>
#include <vector>
#include <string>
#include <tuple>

#include "spherepix/image.h"

namespace spherepix {

/**
 * \brief Pixelation modes
 */
enum PixelationMode {

    //! 1 checker face per cube face.
    MODE_1 = 1,

    //! 2 horizontal checker faces per cube face.
    MODE_2 = 2,

    //! 4 checker faces per cube face.
    MODE_3 = 3
};

enum FaceNeighbor {
    LEFT = 0,
    TOP = 1,
    RIGHT = 2,
    BOTTOM = 3
};

/**
 * \brief Pixelation class
 *
 * Objects of this class hold the 3D coordinates that form the sphere
 * pixelation and all related attributes needed to perform computations on it.
 */
class Pixelation {

public:
    Pixelation();
    Pixelation(PixelationMode mode, const std::vector<Image<float>>& faceCoordinates,
               const std::vector<std::vector<Image<float>>>& betasInterpolation,
               const std::vector<std::vector<Image<float>>>& etasInterpolation,
               const int interpolationBeltWidth);

    ~Pixelation();

    PixelationMode mode() const;

    int faceHeight() const;
    int faceWidth() const;

    /**
     * \brief returns belt interpolation width in pixels.
     */
    int interpolationBeltWidth() const;

    int faceCount() const;

    const Image<int> faceConnectivityGraph() const;

    const Image<float> faceCoordinates(const int faceIndex);

    const Image<float> interpolationCoordinates(const int faceIndex, FaceNeighbor neighbor) const;

    const Image<float> sphericalInterpolationCoordinates(const int faceIndex, FaceNeighbor neighbor) const;

private:
    PixelationMode __mode;
    std::size_t __faceHeight;
    std::size_t __faceWidth;

    //! grids with the (x, y, z) coordinates.
    std::vector<Image<float> > __coordinates;

    //! width in pixels of the interpolation belt
    int __beltWidth;

    //! interpolation belts for each face of the pixelation
    std::vector<std::vector<Image<float> > > __betasBeltCoordinates;
    std::vector<std::vector<Image<float> > > __etasBeltCoordinates;


    //! face connectivity graph. It depends on the pixelation mode.
    Image<int> __faceConnectivityGraph;
};


class PixelationFactory {

public:
    static Pixelation createPixelation(PixelationMode mode,
                                       const int N,
                                       const int interpolationBeltWidth,
                                       const int springIterations = 1000,
                                       const float extraSeparation = 4.0f,
                                       const float M3_theta = 0.01f * M_PI,
                                       const float dt = 0.2f,
                                       const float M = 1.0f,
                                       const float C = 0.05f,
                                       const float K = 5.0f
                                      );

    static Pixelation createPixelation(PixelationMode mode,
                                       std::vector<Image<float>>& faceList,
                                       const int interpolationBeltWidth);

    static Pixelation load(const std::string& path);

    static void save(const Pixelation& pix);

};


template<typename T>
class SphericalImage {

public:
    SphericalImage() {

    }

    SphericalImage(const Pixelation& pix, const int depth = 1):
        __imageList(pix.faceCount()) {

        __faceHeight = pix.faceHeight();
        __faceWidth = pix.faceWidth();
        __depth = depth;
        __mode = pix.mode();

        // allocate memory for the image faces
        for (int i = 0; i < pix.faceCount(); i ++) {
            __imageList[i] = Image<T>(__faceHeight, __faceWidth, __depth);
            __imageList[i].clear();
        }

    }

    inline int faceHeight() const {return __faceHeight; }
    inline int faceWidth() const {return __faceWidth; }
    inline int depth() const { return __depth; }
    int faceCount() const { return __imageList.size(); }

    void clear() {
        for(auto face : __imageList) { face.clear(); }
    }

    inline Image<T>& operator[](std::size_t idx) {
        return __imageList[idx];
    }

    inline const Image<T>& operator[](std::size_t idx) const {
        return __imageList[idx];
    }


private:
    std::vector<Image<T>> __imageList;
    PixelationMode __mode;
    std::size_t __faceWidth;
    std::size_t __faceHeight;
    std::size_t __depth;
};


//#########################################################
// CONNECTIVITY GRAPHS ARRAYS
//#########################################################
extern Image<int> FACE_CONNECTIVITY_MODE_1;
extern Image<int> FACE_CONNECTIVITY_MODE_2;
extern Image<int> FACE_CONNECTIVITY_MODE_3;

extern int __CONNECTIVITY_GRAPH_MODE_1[];
extern int __CONNECTIVITY_GRAPH_MODE_2[];
extern int __CONNECTIVITY_GRAPH_MODE_3[];

//#########################################################
// MODULE METHODS
//#########################################################


/**
 * \brief returns the separation between two pixels in a grid
 *  of spherical coordinates
 *
 * The separation is computed as the distance between two
 * points using orthogonal retraction. The reference points
 * are chosen to be at the center of the grid (height/2, width/2)
 */
float pixelSeparation(const Image<float>& etas);

/**
 * \brief Creates an equidistant pixelation of one cube face to the sphere
 *
 * \param N face side
 *
 * \return Image<float> of shape [N, N, 3] with the spherical coordinates
 *      (x, y, z) normalized.
 */
Image<float> createFace_equidistant(const int N);

/**
 * \brief Creates an equiangular pixelation of one cube face to the sphere
 *
 * \param N face side
 *
 * \return Image<float> of shape [N, N, 3] with the spherical coordinates
 *      (x, y, z) normalized.
 */
Image<float> createFace_equiangular(const int N);

/**
 * \brief Regularizes spherical coordinates grid etas according
 *      to the the mass-spring system.
 *
 * \param etas spherical coordinates grid. (x, y, z) normalized.
 *
 * \param Ls additional separation factor, in pixels, added to the resting
 *      elongation of the springs.
 *      The total resting elongation is given by:
 *      L = L + (Ls/N)*L
 *      where N is the side lenght of the face
 *
 * \param I number of iterations to run the mass-spring system.
 *
 * \param dt time step for spring system simulation.
 *
 * \param M mass of each point.
 *
 * \param C velocity damping factor.
 *
 * \param K spring elastic constant.
 */
Image<float> regularizeCoordinates(const Image<float>& etas,
                                   const float Ls = 4.0f,
                                   const int I = 1000,
                                   const float dt = 0.2f,
                                   const float M = 1.0f,
                                   const float C = 0.05f,
                                   const float K = 5.0f);

/**
 * \brief Creates a face of the sphere pixelation.
 *
 * \param mode pixelation mode.
 *
 * \param N face side length, in pixels. It must be an even number.
 *
 * \param I number of iterations to run the mass-spring system
 *
 * \param Ls additional separation factor, in pixels, added to the resting
 *      elongation of the springs.
 *      The total resting elongation is given by:
 *      L = L + (Ls/N)*L
 *      where N is the side lenght of the face
 *
 * \param M3_theta rotation angle, in radians, between oposite checker faces in MODE_3
  *     of the pixelation. Default to 0.01*PI.
 */
std::vector<Image<float> > createFace(PixelationMode mode,
                                      const int N,
                                      const int I = 1000,
                                      const float Ls = 4.0f,
                                      const float M3_theta = 0.01 * M_PI,
                                      const float dt = 0.2f,
                                      const float M = 1.0f,
                                      const float C = 0.05f,
                                      const float K = 5.0f
                                     );

/**
 * \brief Creates the cube pixelation of the sphere by rotating the input face
 *      to all 6 sides of the cube
 *
 * \param face list of checker faces that makes one face of the cube.
 *      \see{createFace}
 */
std::vector<Image<float> > createCubeFaces(std::vector<Image<float> >& face);


/**
 * \brief Returns the face connectivity graph given a pixelation mode.
 *
 *            1
 *  --------------------
 *       |        |
 *     0 |  face  |  2
 *       |        |
 *  --------------------
 *            3
 *
 * \returns 2D array. Each row contains the connections on the 4 sides of a face.
 */
const Image<int> getFaceConnectivityGraph(PixelationMode mode);


/**
 * \brief returns the face interpolation belts for a given face.
 *
 *            1
 *  --------------------
 *       |        |
 *     0 |  face  |  2
 *       |        |
 *  --------------------
 *            3
 *
 * \param faceList list of spherical coordinates that make the pixelation.
 * \param faceIndex face index for which the interpolation belts are created.
 * \param beltWidth width, in pixels, of the interpolation belt
 * \param mode pixelation mode
 *
 * \returns vector containing the interpolation belts. Each element is a tuple
 *      (betas, etas) where betas are the pixel interpolated coordinates (row, col)
 *      etas contains the respective spherical coordinates.
 *
 */
std::vector<std::pair<Image<float>, Image<float>>> faceInterpolationBelts(std::vector<Image<float>>& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode);


/**
 * \brief computes interpolation belt zero
 *
 * \returns tuple (betas, etas) with the interpolated pixel coordinates and
 *      spherical coordinates.
 */
std::pair<Image<float>, Image<float>> interpolationBelt_0(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode);

/**
 * \brief computes interpolation belt one
 *
 * \returns tuple (betas, etas) with the interpolated pixel coordinates and
 *      spherical coordinates.
 */
std::pair<Image<float>, Image<float>> interpolationBelt_1(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode);

/**
 * \brief computes interpolation belt two
 *
 * \returns tuple (betas, etas) with the interpolated pixel coordinates and
 *      spherical coordinates.
 */
std::pair<Image<float>, Image<float>> interpolationBelt_2(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode);

/**
 * \brief computes interpolation belt three
 *
 * \returns tuple (betas, etas) with the interpolated pixel coordinates and
 *      spherical coordinates.
 */
std::pair<Image<float>, Image<float>> interpolationBelt_3(std::vector<Image<float>>& faceList,
                                   const int faceIndex,
                                   const int beltWidth,
                                   PixelationMode mode);


//#########################################################
// CONVOLUTION METHODS
//#########################################################

template<typename T>
SphericalImage<T> convolve2D(const Pixelation& pix,
                             const SphericalImage<T>& img,
                             const Image<T>& mask);

template<typename T>
SphericalImage<T> convolveRow(const Pixelation& pix,
                              const SphericalImage<T>& img,
                              const Image<T>& mask);

template<typename T>
SphericalImage<T> convolveColumn(const Pixelation& pix,
                                 const SphericalImage<T>& img,
                                 const Image<T>& mask);

template<typename T>
void convolveFace2D(const Pixelation& pix, const SphericalImage<T>& sphericalImg,
                    const int faceIndex, const Image<T>& mask, Image<T>& output);

template<typename T>
void convolveFaceRow(const Pixelation& pix, const SphericalImage<T>& sphericalImg,
                     const int faceIndex, const Image<T>& mask, Image<T>& output);

template<typename T>
void convolveFaceColumn(const Pixelation& pix, const SphericalImage<T>& sphericalImg,
                        const int faceIndex, const Image<T>& mask, Image<T>& output);


//#########################################################
// COORDINATE CASTING
//#########################################################

Image<float> castCoordinates(const Image<float>& etas_0,
  const Image<float>& etas_1, const bool flipVertical=true);


//#########################################################
// include the implementation of template methods
//#########################################################
#include "spherepix/pixelation_impl.h"

}; // namespace spherepix

#endif // SPHEREPIX_PIXELATION_H_