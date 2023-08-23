#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_GRAD_IMPL_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_GRAD_IMPL_H

#include <type_traits>
#include <utility>

#include "exp/operator/constant.h"
#include "exp/operator/conv.h"
#include "exp/operator/log_softmax.h"
#include "exp/operator/nll_loss.h"
#include "exp/operator/reduce_op.h"
#include "utils/fixed_array.h"
#include "utils/base_config.h"
#include "utils/exception.h"

namespace KD {
template <typename ImplType>
class GradImpl {
 public:
  const ImplType &Self() const { return *static_cast<ImplType *>(this); }
};

template <typename OperatorType, typename GIType, typename OIType>
typename std::enable_if<OperatorType::AllowBroadcast::value, IndexArray>::type
GradResultSize(const GIType &grad, const OIType &) {
  return grad.GradSize();
}

template <typename OperatorType, typename GIType, typename OIType>
typename std::enable_if<!OperatorType::AllowBroadcast::value, IndexArray>::type
GradResultSize(const GIType &, const OIType &operand) {
  return static_cast<const IndexArray &>(operand.Size());
}

template <typename OperatorType, typename GIType, typename LhsType,
          typename RhsType>
typename std::enable_if<OperatorType::AllowBroadcast::value, IndexArray>::type
GradResultSize(const GIType &grad, const LhsType &lhs, const RhsType &rhs) {
  CHECK_EQUAL(lhs.DimensionsSize(), rhs.DimensionsSize(),
              "Backward of broadcasting is supported only when the dimensions "
              "of operands are equal, but got "
                  << lhs.DimensionsSize() << " and " << rhs.DimensionsSize()
                  << ".");
  return grad.GradSize();
}

template <typename OperatorType, typename GIType, typename LhsType,
          typename RhsType>
typename std::enable_if<!OperatorType::AllowBroadcast::value &&
                            OperatorType::IsLhs::value,
                        IndexArray>::type
GradResultSize(const GIType &, const LhsType &lhs, const RhsType &) {
  return static_cast<const IndexArray &>(lhs.Size());
}

template <typename OperatorType, typename GIType, typename LhsType,
          typename RhsType>
typename std::enable_if<!OperatorType::AllowBroadcast::value &&
                            OperatorType::IsRhs::value,
                        IndexArray>::type
GradResultSize(const GIType &, const LhsType &, const RhsType &rhs) {
  return static_cast<const IndexArray &>(rhs.Size());
}

template <typename OperatorType, typename GIType, typename OIType>
class UnaryGradImpl
    : public GradImpl<UnaryGradImpl<OperatorType, GIType, OIType>> {
 public:
  UnaryGradImpl(const GIType &grad, const OIType &operand)
      : grad_(grad), operand_(operand) {}

  IndexArray GradSize() const {
    return GradResultSize<OperatorType, GIType, OIType>(grad_, operand_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return OperatorType::Map(indexes, grad_, operand_);
  }

 private:
  const GIType &grad_;
  const OIType &operand_;
};

template <typename OperatorType, typename GIType, typename LhsImplType,
          typename RhsImplType>
class BinaryGradImpl
    : public GradImpl<
          BinaryGradImpl<OperatorType, GIType, LhsImplType, RhsImplType>> {
 public:
  BinaryGradImpl(const GIType &grad, const LhsImplType &lhs,
                 const RhsImplType &rhs)
      : grad_(grad), lhs_(lhs), rhs_(rhs) {}

  IndexArray GradSize() const {
    return GradResultSize<OperatorType, GIType, LhsImplType, RhsImplType>(
        grad_, lhs_, rhs_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return OperatorType::Map(indexes, grad_, lhs_, rhs_);
  }

 private:
  const GIType &grad_;
  const LhsImplType &lhs_;
  const RhsImplType &rhs_;
};

template <typename GIType, typename OIType>
class UnaryGradImpl<typename Operator::LogSoftmax::Grad, GIType, OIType>
    : public GradImpl<
          UnaryGradImpl<typename Operator::LogSoftmax::Grad, GIType, OIType>> {
 public:
  UnaryGradImpl(const GIType &grad, const OIType &operand,
                BasicData *batch_sum_exp, BasicData *batch_max_cls)
      : grad_(grad),
        operand_(operand),
        batch_sum_exp_(batch_sum_exp),
        batch_max_cls_(batch_max_cls) {}

  IndexArray GradSize() const {
    return GradResultSize<typename Operator::LogSoftmax::Grad, GIType, OIType>(
        grad_, operand_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::LogSoftmax::Grad::Map(indexes, grad_, operand_,
                                           batch_sum_exp_, batch_max_cls_);
  }

 private:
  const GIType &grad_;
  const OIType &operand_;

  BasicData *batch_sum_exp_;
  BasicData *batch_max_cls_;
};

template <typename GIType, typename OIType>
class UnaryGradImpl<typename Operator::Mean::Grad, GIType, OIType>
    : public GradImpl<
          UnaryGradImpl<typename Operator::Mean::Grad, GIType, OIType>> {
 public:
  UnaryGradImpl(const GIType &grad, const OIType &operand, Index reduce_dim)
      : grad_(grad), operand_(operand), reduce_dim_(reduce_dim) {}

  IndexArray GradSize() const {
    return GradResultSize<typename Operator::Mean::Grad, GIType, OIType>(
        grad_, operand_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Mean::Grad::Map(indexes, grad_, operand_, reduce_dim_);
  }

 private:
  const GIType &grad_;
  const OIType &operand_;

  Index reduce_dim_;
};

template <typename GIType, typename OIType>
class UnaryGradImpl<typename Operator::Max::Grad, GIType, OIType>
    : public GradImpl<
          UnaryGradImpl<typename Operator::Max::Grad, GIType, OIType>> {
 public:
  UnaryGradImpl(const GIType &grad, const OIType &operand, Index reduce_dim)
      : grad_(grad), operand_(operand), reduce_dim_(reduce_dim) {}

  IndexArray GradSize() const {
    return GradResultSize<typename Operator::Max::Grad, GIType, OIType>(
        grad_, operand_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Max::Grad::Map(indexes, grad_, operand_, reduce_dim_);
  }

 private:
  const GIType &grad_;
  const OIType &operand_;

  Index reduce_dim_;
};

template <typename GIType, typename OIType>
class UnaryGradImpl<typename Operator::NLLLoss::Grad, GIType, OIType>
    : public GradImpl<
          UnaryGradImpl<typename Operator::NLLLoss::Grad, GIType, OIType>> {
 public:
  UnaryGradImpl(const GIType &grad, const OIType &operand,
                const Index *batch_label)
      : grad_(grad), operand_(operand), batch_label_(batch_label) {}

  IndexArray GradSize() const {
    return GradResultSize<typename Operator::NLLLoss::Grad, GIType, OIType>(
        grad_, operand_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::NLLLoss::Grad::Map(indexes, grad_, operand_, batch_label_);
  }

 private:
  const GIType &grad_;
  const OIType &operand_;

  const Index *batch_label_;
};

template <typename GIType, typename OIType>
class UnaryGradImpl<typename Operator::Img2col::Grad, GIType, OIType>
    : public GradImpl<
          UnaryGradImpl<typename Operator::Img2col::Grad, GIType, OIType>> {
 public:
  using MatrixSize = typename Operator::Img2col::MatrixSize;

  UnaryGradImpl(const GIType &grad, const OIType &operand,
                MatrixSize kernel_size, MatrixSize stride_size,
                MatrixSize padding_size, MatrixSize out_size)
      : grad_(grad),
        operand_(operand),
        kernel_size_(std::move(kernel_size)),
        stride_size_(std::move(stride_size)),
        padding_size_(std::move(padding_size)),
        out_size_(std::move(out_size)) {}

  IndexArray GradSize() const {
    return GradResultSize<typename Operator::Img2col::Grad, GIType, OIType>(
        grad_, operand_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Img2col::Grad::Map(indexes, grad_, operand_, kernel_size_,
                                        stride_size_, padding_size_, out_size_);
  }

 private:
  const GIType &grad_;
  const OIType &operand_;

  MatrixSize kernel_size_;
  MatrixSize stride_size_;
  MatrixSize padding_size_;
  MatrixSize out_size_;
};

template <>
class UnaryGradImpl<Operator::Constant, void, BasicData>
    : public GradImpl<UnaryGradImpl<Operator::Constant, void, BasicData>> {
 public:
  UnaryGradImpl(BasicData value, IndexArray shape)
      : value_(value), shape_(std::move(shape)) {}

  IndexArray GradSize() const { return shape_; }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Constant::Map(indexes, value_);
  }

 private:
  BasicData value_;
  IndexArray shape_;
};
}  // namespace KD
#endif