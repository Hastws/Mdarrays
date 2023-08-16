#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_CONSTANT_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_CONSTANT_H

#include <type_traits>

#include "utils/base_config.h"
#include "utils/exception.h"

namespace KD {
namespace Operator {
// A paradigm
struct Constant {
  static Index DimensionsSize() { return 1; }

  static Index Size(Index) { return 1; }

  static BasicData Map(IndexArray &, BasicData value) { return value; }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    static BasicData Map(IndexArray &, BasicData) {
      THROW_ERROR("NotImplementError");
    }
  };
};
}  // namespace Operator
}  // namespace KD
#endif