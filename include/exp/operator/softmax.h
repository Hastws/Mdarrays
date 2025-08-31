#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_SOFTMAX_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_SOFTMAX_H

#include <cmath>
#include <memory>
#include <type_traits>

#include "utils/base_config.h"

namespace Autoalg {
namespace Operator {

// 与 LogSoftmax 保持同样的 2D 约定：输入形状 (batch, classes)
struct Softmax {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 2;
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand) {
    return operand.Size(idx);
  }

  // y[b, j] = exp(x[b, j] - max_cls[b]) / sum_exp[b]
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       const BasicData *batch_sum_exp,
                       const BasicData *batch_max_cls) {
    const Index b = indexes[0];
    const BasicData x = operand.Eval(indexes);
    return std::exp(x - batch_max_cls[b]) / batch_sum_exp[b];
  }

  // 逐 batch 预计算 max 与 sum(exp(x - max))
  template <typename OperandType>
  static void Precompute(const OperandType &operand, BasicData *batch_sum_exp,
                         BasicData *batch_max_cls) {
    const Index n_batch = operand.Size(0);
    const Index n_class = operand.Size(1);

    auto row_ptr =
        Allocator::SharedAllocate<BasicData>(n_class * sizeof(BasicData));
    auto row = row_ptr.get();

    IndexArray idx(2);
    for (Index b = 0; b < n_batch; ++b) {
      idx[0] = b;

      BasicData max_v = DATA_MIN;
      for (Index j = 0; j < n_class; ++j) {
        idx[1] = j;
        row[j] = operand.Eval(idx);
        if (row[j] > max_v) max_v = row[j];
      }
      BasicData sum_e = 0;
      for (Index j = 0; j < n_class; ++j) {
        sum_e += std::exp(row[j] - max_v);
      }
      batch_max_cls[b] = max_v;
      batch_sum_exp[b] = sum_e;
    }
  }

  // 反向： dL/dx_j = s_j * ( g_j - sum_i g_i * s_i )
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand,
                         const BasicData *batch_sum_exp,
                         const BasicData *batch_max_cls) {
      const Index b = indexes[0];
      const Index n_cls = operand.Size(1);

      // s_j
      const BasicData xj = operand.Eval(indexes);
      const BasicData sj = std::exp(xj - batch_max_cls[b]) / batch_sum_exp[b];

      // dot = Σ_i g_i * s_i
      BasicData dot = 0;
      IndexArray i_idx(indexes);
      for (Index i = 0; i < n_cls; ++i) {
        i_idx[1] = i;
        const BasicData xi = operand.Eval(i_idx);
        const BasicData si = std::exp(xi - batch_max_cls[b]) / batch_sum_exp[b];
        dot += grad.Eval(i_idx) * si;
      }
      return sj * (grad.Eval(indexes) - dot);
    }
  };
};

}  // namespace Operator
}  // namespace Autoalg
#endif
