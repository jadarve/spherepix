/**
 * arithmetic.h
 */

#ifndef SPHEREPIX_ARITHMETIC_H_
#define SPHEREPIX_ARITHMETIC_H_

#include <limits>
#include <cmath>

#include "Eigen/Dense"

namespace spherepix {

bool areEqual(const float& x1, const float& x2,
              const float& tolerance = std::numeric_limits<float>::min());

bool isZero(const float& x,
            const float& tolerance = std::numeric_limits<float>::min());


// template<typename Derived>
// bool isNaN(const Eigen::BaseDense<Derived>& x)
// {
//     // return x.array() != x.array();
//     return x != x;
// }

template<typename T>
bool isNaN(const T& x) {
    return x != x;
}

// bool isNaN(const Eigen::Vector3f& v);

}; // namespace spherepix

#endif // SPHEREPIX_ARITHMETIC_H_