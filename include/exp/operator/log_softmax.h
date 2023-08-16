#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_LOG_SOFTMAX_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_LOG_SOFTMAX_H

#include <cmath>
#include <memory>
#include <type_traits>

#include "utils/base_config.h"

namespace KD {
namespace Operator {
// 1. It is easier to obtain derivatives during logarithmic operations, which
// speeds up the speed of backpropagation.
// 2. Solve the possible overflow and underflow problems of Softmax.
struct LogSoftmax {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 2;
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand) {
    return operand.Size(idx);
  }

  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       BasicData *batch_sum_exp, BasicData *batch_max_cls) {
    BasicData value = operand.Eval(indexes);
    return value - batch_max_cls[indexes[0]] -
           std::log(batch_sum_exp[indexes[0]]);
  }

  template <typename OperandType>
  static void Precompute(const OperandType &operand, BasicData *batch_sum_exp,
                         BasicData *batch_max_cls) {
    Index n_batch = operand.Size(0);
    Index n_class = operand.Size(1);
    auto batch_ptr =
        Allocator::SharedAllocate<BasicData>(n_class * sizeof(BasicData));
    auto batch = batch_ptr.get();
    IndexArray indexes(2);

    for (Index i = 0; i < n_batch; ++i) {
      indexes[0] = i;
      BasicData max_cls = DATA_MIN;

      // Calculate the maximum value of each row of data
      for (Index j = 0; j < n_class; ++j) {
        indexes[1] = j;
        batch[j] = operand.Eval(indexes);
        if (batch[j] > max_cls) {
          max_cls = batch[j];
        }
      }

      BasicData sum_exp = 0;
      for (KD::Index j = 0; j < n_class; ++j) {
        sum_exp += std::exp(batch[j] - max_cls);
      }

      batch_sum_exp[i] = sum_exp;
      batch_max_cls[i] = max_cls;
    }
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand,
                         const BasicData *batch_sum_exp,
                         BasicData *batch_max_cls) {
      BasicData x = operand.Eval(indexes);
      BasicData softmax =
          (std::exp(x - batch_max_cls[indexes[0]])) / batch_sum_exp[indexes[0]];

      Index n_cls = operand.Size(1);
      IndexArray grad_indexes(indexes);

      BasicData total_grad = 0;
      Index i = 0;
      for (; i < indexes[1]; ++i) {
        grad_indexes[1] = i;
        total_grad -= grad.Eval(grad_indexes);
      }
      for (++i; i < n_cls; ++i) {
        grad_indexes[1] = i;
        total_grad -= grad.Eval(grad_indexes);
      }
      total_grad *= softmax;

      grad_indexes[1] = indexes[1];
      total_grad += (1 - softmax) * grad.Eval(grad_indexes);
      return total_grad;
    }
  };
};
}  // namespace Operator
}  // namespace KD
#endif