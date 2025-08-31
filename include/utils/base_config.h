#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_BASE_CONFIG_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_BASE_CONFIG_H

#include <limits>

namespace Autoalg {

using Index = unsigned int;
using BasicData = double;

template <typename DataType>
class FixedArray;

using IndexArray = FixedArray<Index>;
constexpr Index FixedArraySize = 8;

constexpr Index INDEX_MAX = std::numeric_limits<Index>::max() >> 1;
constexpr Index INDEX_MIN = 0;
constexpr BasicData DATA_MAX = std::numeric_limits<BasicData>::max();
constexpr BasicData DATA_MIN = std::numeric_limits<BasicData>::lowest();

constexpr BasicData EPS = 1e-12;

}  // namespace Autoalg
#endif