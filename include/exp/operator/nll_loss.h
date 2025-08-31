#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_NLL_LOSS_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_NLL_LOSS_H

#include "utils/base_config.h"

namespace Autoalg {
namespace Operator {
struct NLLLoss {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 1;
  }

  template <typename OperandType>
  static Index Size(Index, const OperandType &operand) {
    return operand.Size(0);
  }

  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       const Index *batch_label) {
    Index idx = indexes[0];
    Index label = batch_label[idx];
    IndexArray operand_indexes{idx, label};
    return -operand.Eval(operand_indexes);
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &, const Index *batch_label) {
      Index idx = indexes[0];
      Index cls = indexes[1];
      if (cls == batch_label[idx]) {
        return -grad.Eval(indexes);
      } else {
        return 0;
      }
    }
  };
};

}  // namespace Operator
}  // namespace Autoalg

#endif