#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_EXP_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_EXP_H

#include "exp/exp_impl.h"
#include "exp/operator/basic_op.h"

namespace Autoalg {
template <typename ImplType>
struct Exp {
 public:
  explicit Exp(Allocator::UniquePtr<ImplType> &&ptr)
      : impl_ptr_(std::move(ptr), false) {}

  const ExpImplPtr<ImplType> &ImplPtr() const { return impl_ptr_; }

  const ImplType &Impl() const { return *impl_ptr_; }

 protected:
  ExpImplPtr<ImplType> impl_ptr_;
};

// Forward declaration and using declaration
namespace Operator {
template <typename OIType>
Exp<UnaryExpImpl<Minus, OIType>> operator-(const Exp<OIType> &operand);

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>> operator+(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs);

template <typename LhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, UnaryExpImpl<Constant, BasicData>>>
operator+(const Exp<LhsImplType> &lhs, BasicData data);

template <typename RhsImplType>
Exp<BinaryExpImpl<Add, UnaryExpImpl<Constant, BasicData>, RhsImplType>>
operator+(BasicData data, const Exp<RhsImplType> &rhs);

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>> operator*(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs);

template <typename LhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, UnaryExpImpl<Constant, BasicData>>>
operator*(const Exp<LhsImplType> &lhs, BasicData data);

template <typename RhsImplType>
Exp<BinaryExpImpl<Mul, UnaryExpImpl<Constant, BasicData>, RhsImplType>>
operator*(BasicData data, const Exp<RhsImplType> &rhs);

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>> operator-(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs);

template <typename LhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, UnaryExpImpl<Constant, BasicData>>>
operator-(const Exp<LhsImplType> &lhs, BasicData data);

template <typename RhsImplType>
Exp<BinaryExpImpl<Sub, UnaryExpImpl<Constant, BasicData>, RhsImplType>>
operator-(BasicData data, const Exp<RhsImplType> &rhs);
}  // namespace Operator

using Operator::operator+;
using Operator::operator*;
using Operator::operator-;

}  // namespace Autoalg

#endif