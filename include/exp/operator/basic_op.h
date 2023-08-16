#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_BASIC_OP_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_BASIC_OP_H

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "utils/base_config.h"

namespace KD {
namespace Operator {

struct UnaryBasicOperator {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType &operand) {
    return operand.DimensionsSize();
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand) {
    return operand.Size(idx);
  }
};

struct BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static Index DimensionsSize(const LhsType &lhs, const RhsType &rhs) {
    return std::max(lhs.DimensionsSize(), rhs.DimensionsSize());
  }

  template <typename LhsType, typename RhsType>
  static Index Size(Index idx, const LhsType &lhs, const RhsType &rhs) {
    if (idx >= lhs.DimensionsSize()) {
      return rhs.Size(idx);
    }
    if (idx >= rhs.DimensionsSize()) {
      return lhs.Size(idx);
    }
    return std::max(lhs.Size(idx), rhs.Size(idx));
  }
};

struct Minus : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand) {
    return -operand.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &) {
      return -grad.Eval(indexes);
    }
  };
};

struct Add : public BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray &indexes, const LhsType &lhs,
                       const RhsType &rhs) {
    return lhs.Eval(indexes) + rhs.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &) {
        return grad.Eval(indexes);
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &) {
        return grad.Eval(indexes);
      }
    };
  };
};

struct Mul : public BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray &indexes, const LhsType &lhs,
                       const RhsType &rhs) {
    return lhs.Eval(indexes) * rhs.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &rhs) {
        return grad.Eval(indexes) * rhs.Eval(indexes);
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &lhs, const RhsType &) {
        return grad.Eval(indexes) * lhs.Eval(indexes);
      }
    };
  };
};

struct Sub : public BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray &indexes, const LhsType &lhs,
                       const RhsType &rhs) {
    return lhs.Eval(indexes) - rhs.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &) {
        return grad.Eval(indexes);
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray &indexes, const GradType &grad,
                           const LhsType &, const RhsType &) {
        return -grad.Eval(indexes);
      }
    };
  };
};

struct ReLU : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand) {
    return std::max(operand.Eval(indexes), BasicData(0));
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand) {
      return operand.Eval(indexes) > BasicData(0) ? grad.Eval(indexes)
                                                  : BasicData(0);
    }
  };
};

struct Sigmoid : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand) {
    return 1 / (1 + std::exp(-operand.Eval(indexes)));
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand) {
      BasicData value = Sigmoid::Map(indexes, operand);
      return value * (1 - value) * grad.Eval(indexes);
    }
  };
};

struct Identity : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand) {
    return operand.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &) {
      return grad.Eval(indexes);
    }
  };
};
}  // namespace Operator
}  // namespace KD
#endif