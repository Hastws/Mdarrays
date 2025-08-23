#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_BASIC_OP_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_BASIC_OP_H

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "utils/base_config.h"
#include "utils/utils.h"

namespace KD {
namespace Operator {

struct UnaryBasicOperator {
  template <typename OperandType>
  static Index DimensionsSize(const OperandType& operand) {
    return operand.DimensionsSize();
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType& operand) {
    return operand.Size(idx);
  }
};

struct BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static Index DimensionsSize(const LhsType& lhs, const RhsType& rhs) {
    return std::max(lhs.DimensionsSize(), rhs.DimensionsSize());
  }

  template <typename LhsType, typename RhsType>
  static Index Size(Index idx, const LhsType& lhs, const RhsType& rhs) {
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
  static BasicData Map(IndexArray& indexes, const OperandType& operand) {
    return -operand.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& indexes, const GradType& grad,
                         const OperandType&) {
      return -grad.Eval(indexes);
    }
  };
};

struct Add : public BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray& indexes, const LhsType& lhs,
                       const RhsType& rhs) {
    return lhs.Eval(indexes) + rhs.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray& indexes, const GradType& grad,
                           const LhsType&, const RhsType&) {
        return grad.Eval(indexes);
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray& indexes, const GradType& grad,
                           const LhsType&, const RhsType&) {
        return grad.Eval(indexes);
      }
    };
  };
};

struct Mul : public BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray& indexes, const LhsType& lhs,
                       const RhsType& rhs) {
    return lhs.Eval(indexes) * rhs.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray& indexes, const GradType& grad,
                           const LhsType&, const RhsType& rhs) {
        return grad.Eval(indexes) * rhs.Eval(indexes);
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray& indexes, const GradType& grad,
                           const LhsType& lhs, const RhsType&) {
        return grad.Eval(indexes) * lhs.Eval(indexes);
      }
    };
  };
};

struct Sub : public BinaryBasicOperator {
  template <typename LhsType, typename RhsType>
  static BasicData Map(IndexArray& indexes, const LhsType& lhs,
                       const RhsType& rhs) {
    return lhs.Eval(indexes) - rhs.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;

    struct Lhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::true_type;
      using IsRhs = std::false_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray& indexes, const GradType& grad,
                           const LhsType&, const RhsType&) {
        return grad.Eval(indexes);
      }
    };

    struct Rhs {
      using AllowBroadcast = Grad::AllowBroadcast;
      using IsLhs = std::false_type;
      using IsRhs = std::true_type;

      template <typename GradType, typename LhsType, typename RhsType>
      static BasicData Map(IndexArray& indexes, const GradType& grad,
                           const LhsType&, const RhsType&) {
        return -grad.Eval(indexes);
      }
    };
  };
};

struct ReLU : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& indexes, const OperandType& operand) {
    return std::max(operand.Eval(indexes), BasicData(0));
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& indexes, const GradType& grad,
                         const OperandType& operand) {
      return operand.Eval(indexes) > BasicData(0) ? grad.Eval(indexes)
                                                  : BasicData(0);
    }
  };
};

struct Sigmoid : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& indexes, const OperandType& operand) {
    return 1 / (1 + std::exp(-operand.Eval(indexes)));
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& indexes, const GradType& grad,
                         const OperandType& operand) {
      BasicData value = Sigmoid::Map(indexes, operand);
      return value * (1 - value) * grad.Eval(indexes);
    }
  };
};

struct Tanh : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    return std::tanh(x.Eval(idx));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData t = std::tanh(x.Eval(idx));
      return (1 - t * t) * g.Eval(idx);
    }
  };
};

struct ExpFunction : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    return std::exp(x.Eval(idx));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData y = std::exp(x.Eval(idx));
      return y * g.Eval(idx);
    }
  };
};

struct Log : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    BasicData v_clamp = std::max(v, EPS);
    return std::log(v_clamp);
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData v_clamp = std::max(v, EPS);
      return g.Eval(idx) / v_clamp;
    }
  };
};

struct Sqrt : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = std::max(x.Eval(idx), EPS);
    return std::sqrt(v);
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = std::max(x.Eval(idx), EPS);
      BasicData s = std::sqrt(v);
      return g.Eval(idx) * (static_cast<BasicData>(0.5) / s);
    }
  };
};

struct Rsqrt : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = std::max(x.Eval(idx), EPS);
    return static_cast<BasicData>(1) / std::sqrt(v);
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = std::max(x.Eval(idx), EPS);
      BasicData y = static_cast<BasicData>(1) / std::sqrt(v);  // v^{-1/2}
      return g.Eval(idx) *
             (static_cast<BasicData>(-0.5) * y * y * y);  // -1/2 v^{-3/2}
    }
  };
};

struct Reciprocal : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    BasicData d = (std::fabs(v) < EPS) ? (v >= 0 ? EPS : -EPS) : v;
    return static_cast<BasicData>(1) / d;
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData d = (std::fabs(v) < EPS) ? (v >= 0 ? EPS : -EPS) : v;
      BasicData inv = static_cast<BasicData>(1) / d;
      return g.Eval(idx) * (-inv * inv);
    }
  };
};

struct Softplus : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    if (v > 0) return v + std::log1p(std::exp(-v));
    return std::log1p(std::exp(v));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData s = static_cast<BasicData>(1) /
                    (static_cast<BasicData>(1) + std::exp(-v));  // sigmoid
      return g.Eval(idx) * s;
    }
  };
};

struct Swish : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    BasicData s =
        static_cast<BasicData>(1) / (static_cast<BasicData>(1) + std::exp(-v));
    return v * s;
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData s = static_cast<BasicData>(1) /
                    (static_cast<BasicData>(1) + std::exp(-v));
      return g.Eval(idx) * (s + v * s * (1 - s));
    }
  };
};

struct Mish : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    BasicData sp =
        (v > 0) ? v + std::log1p(std::exp(-v)) : std::log1p(std::exp(v));
    return v * std::tanh(sp);
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData sp =
          (v > 0) ? v + std::log1p(std::exp(-v)) : std::log1p(std::exp(v));
      BasicData th = std::tanh(sp);
      BasicData s = static_cast<BasicData>(1) /
                    (static_cast<BasicData>(1) + std::exp(-v));  // sigmoid
      BasicData dy =
          th + v * (1 - th * th) * s;  // d/dx tanh(sp)= (1-th^2)*sigmoid(x)
      return g.Eval(idx) * dy;
    }
  };
};

struct GELU : public UnaryBasicOperator {
  static constexpr BasicData kC =
      static_cast<BasicData>(0.7978845608028654);  // sqrt(2/pi)
  static constexpr BasicData kK = static_cast<BasicData>(0.044715);
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    BasicData u = kC * (v + kK * v * v * v);
    BasicData t = std::tanh(u);
    return static_cast<BasicData>(0.5) * v * (static_cast<BasicData>(1) + t);
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData u = kC * (v + kK * v * v * v);
      BasicData t = std::tanh(u);
      BasicData du_dv = kC * (static_cast<BasicData>(1) +
                              static_cast<BasicData>(3) * kK * v * v);
      BasicData dt_dv =
          (static_cast<BasicData>(1) - t * t) * du_dv;  // sech^2 = 1 - tanh^2
      BasicData dy_dv =
          static_cast<BasicData>(0.5) * (static_cast<BasicData>(1) + t) +
          static_cast<BasicData>(0.5) * v * dt_dv;
      return g.Eval(idx) * dy_dv;
    }
  };
};

struct HardSigmoid : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    return Clamp(v / static_cast<BasicData>(6) + static_cast<BasicData>(0.5),
                 static_cast<BasicData>(0), static_cast<BasicData>(1));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      if (v <= static_cast<BasicData>(-3) || v >= static_cast<BasicData>(3))
        return static_cast<BasicData>(0);
      return g.Eval(idx) *
             (static_cast<BasicData>(1) / static_cast<BasicData>(6));
    }
  };
};

struct HardSwish : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    BasicData hs =
        Clamp(v / static_cast<BasicData>(6) + static_cast<BasicData>(0.5),
              static_cast<BasicData>(0), static_cast<BasicData>(1));
    return v * hs;
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      if (v <= static_cast<BasicData>(-3))
        return static_cast<BasicData>(0) * g.Eval(idx);
      if (v >= static_cast<BasicData>(3)) return g.Eval(idx);
      return g.Eval(idx) *
             (v / static_cast<BasicData>(3) + static_cast<BasicData>(0.5));
    }
  };
};

struct Abs : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    return std::fabs(x.Eval(idx));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      if (v > 0) return g.Eval(idx);
      if (v < 0) return -g.Eval(idx);
      return static_cast<BasicData>(0);
    }
  };
};

struct LeakyReLU : public UnaryBasicOperator {
  static constexpr BasicData kAlpha = static_cast<BasicData>(0.01);
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    return (v > 0) ? v : kAlpha * v;
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      return (x.Eval(idx) > 0) ? g.Eval(idx) : (kAlpha * g.Eval(idx));
    }
  };
};

struct ELU : public UnaryBasicOperator {
  static constexpr BasicData kAlpha = static_cast<BasicData>(1);
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    BasicData v = x.Eval(idx);
    return (v >= 0) ? v : kAlpha * (std::exp(v) - static_cast<BasicData>(1));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      if (v >= 0) return g.Eval(idx);
      // d/dx alpha*(e^x - 1) = alpha*e^x = y + alpha
      BasicData y = kAlpha * (std::exp(v) - static_cast<BasicData>(1));
      return g.Eval(idx) * (y + kAlpha);
    }
  };
};

struct ReLU6 : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    return Clamp(x.Eval(idx), static_cast<BasicData>(0),
                 static_cast<BasicData>(6));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      return (v > 0 && v < static_cast<BasicData>(6))
                 ? g.Eval(idx)
                 : static_cast<BasicData>(0);
    }
  };
};

struct Log1p : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    return std::log1p(std::max(x.Eval(idx), -static_cast<BasicData>(1) + EPS));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      BasicData v = x.Eval(idx);
      BasicData d = static_cast<BasicData>(1) +
                    std::max(v, -static_cast<BasicData>(1) + EPS);
      return g.Eval(idx) / d;
    }
  };
};

struct Expm1 : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& idx, const OperandType& x) {
    return std::expm1(x.Eval(idx));
  }
  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;
    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& idx, const GradType& g,
                         const OperandType& x) {
      return g.Eval(idx) * std::exp(x.Eval(idx));
    }
  };
};

struct Identity : public UnaryBasicOperator {
  template <typename OperandType>
  static BasicData Map(IndexArray& indexes, const OperandType& operand) {
    return operand.Eval(indexes);
  }

  struct Grad {
    using AllowBroadcast = std::true_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray& indexes, const GradType& grad,
                         const OperandType&) {
      return grad.Eval(indexes);
    }
  };
};
}  // namespace Operator
}  // namespace KD
#endif