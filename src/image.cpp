
#include "spherepix/image.h"

#include <iostream>

namespace spherepix {

// Instantiate the most used types of images

// template class Image<float>;
// template class Image<int>;


// // Instatiate both implementation of read/write Eigen vectors

// template Eigen::Vector3f readVector3f<true>(const Image<float>& image, const int row, const int col);
// template Eigen::Vector3f readVector3f<false>(const Image<float>& image, const int row, const int col);

// template void writeVector3f<true>(const Eigen::Vector3f& v, const int row, const int col, Image<float>& image);
// template void writeVector3f<false>(const Eigen::Vector3f& v, const int row, const int col, Image<float>& image);

// template Eigen::Vector2f readVector2f<true>(const Image<float>& image, const int row, const int col);
// template Eigen::Vector2f readVector2f<false>(const Image<float>& image, const int row, const int col);

// template void writeVector2f<true>(const Eigen::Vector2f& v, const int row, const int col, Image<float>& image);
// template void writeVector2f<false>(const Eigen::Vector2f& v, const int row, const int col, Image<float>& image);

};