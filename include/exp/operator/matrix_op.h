#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_MATRIX_MUL_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_MATRIX_MUL_H

#include <algorithm>
#include <type_traits>

#include "utils/array.h"
#include "utils/base_config.h"

namespace KD {
namespace Operator {
struct MatrixTranspose {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 2;
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand) {
    return operand.Size(1 - idx);
  }

  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand) {
    std::swap(indexes[0], indexes[1]);
    BasicData value = operand.Eval(indexes);
    std::swap(indexes[0], indexes[1]);
    return value;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &) {
      std::swap(indexes[0], indexes[1]);
      BasicData value = grad.Eval(indexes);
      std::swap(indexes[0], indexes[1]);
      return value;
    }
  };
};

struct MatrixMul {
  template <typename LhsType, typename RhsType>
  static Index DimensionsSize(const LhsType &, const RhsType &) {
    return 2;
  }

  template <typename LhsType, typename RhsType>
  static Index Size(Index idx, const LhsType &lhs, const RhsType &rhs) {
    return idx == 0 ? lhs.Size(0) : rhs.Size(1);
  }

  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray &indexes, const LhsType &lhs,
                       const RhsType &rhs) {
    Index h_size = lhs.Size(1);
    IndexArray lhs_indexes(indexes);
    IndexArray rhs_indexes(indexes);

    BasicData value = 0;
    for (Index i = 0; i < h_size; ++i) {
      lhs_indexes[1] = i;
      rhs_indexes[0] = i;
      value += lhs.Eval(lhs_indexes) * rhs.Eval(rhs_indexes);
    }
    return value;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &rhs) {
        Index h_size = rhs.Size(1);
        IndexArray grad_indexes({indexes[0], 0});
        IndexArray rhs_indexes({indexes[1], 0});

        BasicData value = 0;
        for (Index i = 0; i < h_size; ++i) {
          grad_indexes[1] = i;
          rhs_indexes[1] = i;
          value += grad.Eval(grad_indexes) * rhs.Eval(rhs_indexes);
        }
        return value;
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &lhs, const RhsType &) {
        Index h_size = lhs.Size(0);
        IndexArray lhs_indexes({0, indexes[0]});
        IndexArray grad_indexes({0, indexes[1]});

        BasicData value = 0;
        for (Index i = 0; i < h_size; ++i) {
          lhs_indexes[0] = i;
          grad_indexes[0] = i;
          value += lhs.Eval(lhs_indexes) * grad.Eval(grad_indexes);
        }
        return value;
      }
    };
  };
};

struct BatchMatrixTranspose {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 3;
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand) {
    switch (idx) {
      case 0:
        return operand.Size(0);
      case 1:
        return operand.Size(2);
      default:
        return operand.Size(1);  // case 2
    }
  }

  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand) {
    std::swap(indexes[1], indexes[2]);
    return operand.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &) {
      std::swap(indexes[1], indexes[2]);
      return grad.Eval(indexes);
    }
  };
};

struct BatchMatrixMul {
  template <typename LhsType, typename RhsType>
  static Index DimensionsSize(const LhsType &, const RhsType &) {
    return 3;
  }

  template <typename LhsType, typename RhsType>
  static Index Size(Index idx, const LhsType &lhs, const RhsType &rhs) {
    switch (idx) {
      case 0:
        return std::max(lhs.Size(0), rhs.Size(0));
      case 1:
        return lhs.Size(1);
      default:
        return rhs.Size(2);  // case 2
    }
  }

  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray &indexes, const LhsType &lhs,
                       const RhsType &rhs) {
    Index h_size = lhs.Size(2);
    IndexArray lhs_indexes(indexes);
    IndexArray rhs_indexes(indexes);

    BasicData value = 0;
    for (Index i = 0; i < h_size; ++i) {
      lhs_indexes[2] = i;
      rhs_indexes[1] = i;
      value += lhs.Eval(lhs_indexes) * rhs.Eval(rhs_indexes);
    }
    return value;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &rhs) {
        Index h_size = rhs.Size(2);
        IndexArray grad_indexes({indexes[0], indexes[1], 0});
        IndexArray rhs_indexes({indexes[0], indexes[2], 0});

        BasicData value = 0;
        for (Index i = 0; i < h_size; ++i) {
          grad_indexes[2] = i;
          rhs_indexes[2] = i;
          value += grad.Eval(grad_indexes) * rhs.Eval(rhs_indexes);
        }
        return value;
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &lhs, const RhsType &) {
        Index h_size = lhs.Size(1);
        IndexArray lhs_indexes({indexes[0], 0, indexes[1]});
        IndexArray grad_indexes({indexes[0], 0, indexes[2]});

        BasicData value = 0;
        for (Index i = 0; i < h_size; ++i) {
          lhs_indexes[1] = i;
          grad_indexes[1] = i;
          value += lhs.Eval(lhs_indexes) * grad.Eval(grad_indexes);
        }
        return value;
      }
    };
  };
};
}  // namespace Operator
}  // namespace KD
#endif