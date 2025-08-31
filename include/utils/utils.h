#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_UTILS_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_UTILS_H

#include <algorithm>

#include "utils/base_config.h"

namespace Autoalg {
inline BasicData Clamp(const BasicData v, const BasicData lo,
                       const BasicData hi) {
  return std::max(lo, std::min(v, hi));
}
}  // namespace Autoalg
#endif  // MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_UTILS_H
