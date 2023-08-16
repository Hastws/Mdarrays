#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_REDUCE_OP_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_REDUCE_OP_H

#include <algorithm>

#include "utils/base_config.h"
#include "utils/exception.h"

namespace KD {
namespace Operator {
struct ReduceOperator {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &operand) {
    return std::max(operand.DimensionsSize() - 1, static_cast<Index>(1));
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand, Index reduce_dim) {
    if (operand.DimensionsSize() == 1) {
      return 1;
    } else if (idx < reduce_dim) {
      return operand.Size(idx);
    } else {
      return operand.Size(idx + 1);
    }
  }
};

struct Mean : public ReduceOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       Index reduce_dim) {
    IndexArray operand_indexes(indexes.ArraySize() + 1);
    Index reduce_size = operand.Size(reduce_dim);
    Index index = 0;
    for (; index < reduce_dim; ++index) {
      operand_indexes[index] = indexes[index];
    }
    for (++index; index < indexes.ArraySize() + 1; ++index) {
      operand_indexes[index] = indexes[index - 1];
    }

    BasicData value = 0;
    for (Index i = 0; i < reduce_size; ++i) {
      operand_indexes[reduce_dim] = i;
      value += operand.Eval(operand_indexes);
    }
    value /= reduce_size;
    return value;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand, Index reduce_dim) {
      Index reduce_size = operand.Size(reduce_dim);
      IndexArray grad_indexes(indexes.ArraySize() - 1);
      Index i = 0;
      for (; i < reduce_dim; ++i) {
        grad_indexes[i] = indexes[i];
      }
      for (++i; i < indexes.ArraySize(); ++i) {
        grad_indexes[i - 1] = indexes[i];
      }
      return grad.Eval(grad_indexes) / reduce_size;
    }
  };
};

struct Argmax : public ReduceOperator {
  template <typename OperandType>
  static Index Map(IndexArray &indexes, const OperandType &operand,
                   Index reduce_dim) {
    IndexArray operand_indexes(indexes.ArraySize() + 1);
    Index reduce_size = operand.Size(reduce_dim);
    Index i = 0;
    for (; i < reduce_dim; ++i) {
      operand_indexes[i] = indexes[i];
    }
    for (++i; i < operand_indexes.ArraySize(); ++i) {
      operand_indexes[i] = indexes[i - 1];
    }

    BasicData value, max_value = DATA_MIN;
    Index idx = 0;
    for (i = 0; i < reduce_size; ++i) {
      operand_indexes[reduce_dim] = i;
      value = operand.Eval(operand_indexes);
      if (max_value < value) {
        max_value = value;
        idx = i;
      }
    }
    return idx;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &, const GradType &, const OperandType &,
                         Index) {
      THROW_ERROR("NotImplementError for class Grad in class Argmax.");
    }
  };
};

struct Max : public ReduceOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       Index reduce_dim) {
    IndexArray operand_indexes(indexes.ArraySize() + 1);
    Index reduce_size = operand.Size(reduce_dim);
    Index i = 0;
    for (; i < reduce_dim; ++i) {
      operand_indexes[i] = indexes[i];
    }
    for (++i; i < indexes.ArraySize() + 1; ++i) {
      operand_indexes[i] = indexes[i - 1];
    }

    BasicData value, max_value = DATA_MIN;
    for (operand_indexes[reduce_dim] = 0;
         operand_indexes[reduce_dim] < reduce_size;
         ++operand_indexes[reduce_dim]) {
      value = operand.Eval(operand_indexes);
      max_value = std::max(max_value, value);
    }
    return max_value;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand, Index reduce_dim) {
      Index reduce_size = operand.Size(reduce_dim);
      BasicData key_idx = indexes[reduce_dim];
      BasicData key_value = operand.Eval(indexes);

      for (indexes[reduce_dim] = 0;
           indexes[reduce_dim] < key_idx && operand.Eval(indexes) < key_value;
           ++indexes[reduce_dim]) {
      }

      if (indexes[reduce_dim] != key_idx) {
        return 0;
      }

      for (++indexes[reduce_dim]; indexes[reduce_dim] < reduce_size &&
                                  operand.Eval(indexes) < key_value;
           ++indexes[reduce_dim]) {
      }
      if (indexes[reduce_dim] != reduce_size) {
        return 0;
      }

      Index i = 0;
      IndexArray grad_indexes(indexes.ArraySize() - 1);
      for (; i < reduce_dim; ++i) grad_indexes[i] = indexes[i];
      for (++i; i < indexes.ArraySize(); ++i) grad_indexes[i - 1] = indexes[i];
      return grad.Eval(grad_indexes);
    }
  };
};
}  // namespace Operator
}  // namespace KD

#endif