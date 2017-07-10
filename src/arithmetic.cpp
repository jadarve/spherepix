#include "spherepix/arithmetic.h"

namespace spherepix {

bool areEqual(const float& x1, const float& x2, const float& tolerance) {
    return fabs(x1 - x2) < tolerance;
}

bool isZero(const float& x, const float& tolerance) {
    return fabs(x) < tolerance;
}

// bool isNaN(const Eigen::Vector3f& v) {
//     return v != v;
// }

}; // namespace spherepix