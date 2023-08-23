#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_EXP_IMPL_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_EXP_IMPL_H

#include <initializer_list>
#include <memory>
#include <utility>

#include "exp/grad_impl.h"
#include "exp/operator/constant.h"
#include "exp/operator/conv.h"
#include "exp/operator/log_softmax.h"
#include "exp/operator/nll_loss.h"
#include "exp/operator/reduce_op.h"
#include "memory_pool/allocator.h"
#include "utils/array.h"
#include "utils/base_config.h"

namespace KD {

// Forward declaration
template <typename T>
class ExpImpl;

template <typename T>
class ExpImplPtr;

template <typename OperatorType, typename OIType>
class UnaryExpImpl;

template <typename OperatorType, typename LhsImplType, typename RhsImplType>
class BinaryExpImpl;

template <typename ImplType>
class ExpImpl {
 public:
  Index RefCount() const { return ref_count_; }

  Index GradCount() const { return grad_count_; }

  void IncrementRefCount() { ++ref_count_; };

  void DecreaseRefCount() { --ref_count_; };

  void IncrementGradCount() { ++grad_count_; };

  void DecreaseGradCount() { --grad_count_; };

 private:
  Index ref_count_ = 0;
  Index grad_count_ = 0;
};

template <typename ImplType>
class ExpImplPtr {
 public:
  ExpImplPtr(Allocator::UniquePtr<ImplType> &&ptr, bool with_grad)
      : ptr_(ptr.release()),
        with_grad_(with_grad && static_cast<ImplType *>(ptr_)->RequiresGrad()) {
    IncrementCounters();
  }

  ExpImplPtr(const ImplType &impl, bool with_grad)
      : ptr_(const_cast<ImplType *>(&impl)),
        with_grad_(with_grad && static_cast<ImplType *>(ptr_)->RequiresGrad()) {
    IncrementCounters();
  }

  ExpImplPtr(const ExpImplPtr &other, bool with_grad)
      : ptr_(other.ptr_),
        with_grad_(with_grad && static_cast<ImplType *>(ptr_)->RequiresGrad()) {
    IncrementCounters();
  }

  ~ExpImplPtr() { DecreaseRefcount(); }

  ImplType *operator->() const { return static_cast<ImplType *>(ptr_); }

  const ImplType &operator*() const { return *static_cast<ImplType *>(ptr_); }

  explicit operator bool() const { return ptr_ != nullptr; }

  template <typename GradImplType>
  void InvokeBackward(const GradImplType &grad) {
    auto ptr = static_cast<ImplType *>(ptr_);
    if (ptr->RequiresGrad()) {
      if (with_grad_) {
        ptr_->DecreaseGradCount();
      }
      ptr->Backward(grad);
    }
  }

 private:
  void IncrementCounters() {
    ptr_->IncrementRefCount();
    if (with_grad_) {
      ptr_->IncrementGradCount();
    }
  }

  void DecreaseRefcount() {
    ptr_->DecreaseRefCount();
    if (ptr_->RefCount() == 0) {
      delete_handler(static_cast<void *>(ptr_));
    }
  }

  ExpImpl<ImplType> *ptr_;
  bool with_grad_;
  Allocator::DeleteHandler<ImplType> delete_handler;
};
}  // namespace KD

namespace KD {
template <typename OperatorType, typename OIType>  // OIType = OperandImplType
class UnaryExpImpl : public ExpImpl<UnaryExpImpl<OperatorType, OIType>> {
 public:
  using Operator = OperatorType;
  using operand_type = OIType;

  explicit UnaryExpImpl(const ExpImplPtr<OIType> &ptr)
      : operand_ptr_(ptr, true) {}

  Index DimensionsSize() const {
    return OperatorType::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const { return OperatorType::Size(idx, *operand_ptr_); }

  BasicData Eval(IndexArray &indexes) const {
    return OperatorType::Map(indexes, *operand_ptr_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");
    UnaryGradImpl<typename OperatorType::Grad, GIType, OIType> out_grad(
        grad, *operand_ptr_);
    operand_ptr_.InvokeBackward(out_grad);
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
};

template <typename OperatorType, typename LhsImplType, typename RhsImplType>
class BinaryExpImpl
    : public ExpImpl<BinaryExpImpl<OperatorType, LhsImplType, RhsImplType>> {
 public:
  using Operator = OperatorType;
  using lhs_type = LhsImplType;
  using rhs_type = RhsImplType;

  BinaryExpImpl(const ExpImplPtr<LhsImplType> &lhs_ptr,
                const ExpImplPtr<RhsImplType> &rhs_ptr)
      : lhs_ptr_(lhs_ptr, true), rhs_ptr_(rhs_ptr, true) {}

  Index DimensionsSize() const {
    return OperatorType::DimensionsSize(*lhs_ptr_, *rhs_ptr_);
  }

  Index Size(Index idx) const {
    return OperatorType::Size(idx, *lhs_ptr_, *rhs_ptr_);
  }

  BasicData Eval(IndexArray &indexes) const {
    return OperatorType::Map(indexes, *lhs_ptr_, *rhs_ptr_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  bool RequiresGrad() const {
    return lhs_ptr_->RequiresGrad() || rhs_ptr_->RequiresGrad();
  }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");

    BinaryGradImpl<typename OperatorType::Grad::Lhs, GIType, LhsImplType,
                   RhsImplType>
        lhs_grad(grad, *lhs_ptr_, *rhs_ptr_);
    lhs_ptr_.InvokeBackward(lhs_grad);

    BinaryGradImpl<typename OperatorType::Grad::Rhs, GIType, LhsImplType,
                   RhsImplType>
        rhs_grad(grad, *lhs_ptr_, *rhs_ptr_);
    rhs_ptr_.InvokeBackward(rhs_grad);
  }

 private:
  ExpImplPtr<LhsImplType> lhs_ptr_;
  ExpImplPtr<RhsImplType> rhs_ptr_;
};
}  // namespace KD

namespace KD {

template <typename OIType>
class UnaryExpImpl<Operator::LogSoftmax, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::LogSoftmax, OIType>> {
 public:
  using Operator = Operator::LogSoftmax;
  using operand_type = OIType;

  explicit UnaryExpImpl(const ExpImplPtr<OIType> &ptr)
      : operand_ptr_(ptr, true),
        n_batch_(operand_ptr_->Size(0)),
        batch_sum_exp_(Allocator::UniqueAllocate<BasicData>(
            sizeof(BasicData) * operand_ptr_->Size(0))),
        batch_max_cls_(Allocator::UniqueAllocate<BasicData>(
            sizeof(BasicData) * operand_ptr_->Size(0))) {
    Operator::LogSoftmax::Precompute(*operand_ptr_, batch_sum_exp_.get(),
                                     batch_max_cls_.get());
  }

  Index DimensionsSize() const {
    return Operator::LogSoftmax::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::LogSoftmax::Size(idx, *operand_ptr_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::LogSoftmax::Map(
        indexes, *operand_ptr_, batch_sum_exp_.get(), batch_max_cls_.get());
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");

    UnaryGradImpl<typename Operator::LogSoftmax::Grad, GIType, OIType> out_grad(
        grad, *operand_ptr_, batch_sum_exp_.get(), batch_max_cls_.get());
    operand_ptr_.InvokeBackward(out_grad);
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  Index n_batch_;
  Allocator::UniquePtr<BasicData> batch_sum_exp_;
  Allocator::UniquePtr<BasicData> batch_max_cls_;
};

template <typename OIType>
class UnaryExpImpl<Operator::Mean, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::Mean, OIType>> {
 public:
  using Operator = Operator::Mean;
  using operand_type = OIType;

  explicit UnaryExpImpl(const ExpImplPtr<OIType> &ptr, Index reduce_dim)
      : operand_ptr_(ptr, true), reduce_dim_(reduce_dim) {}

  Index DimensionsSize() const {
    return Operator::Mean::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::Mean::Size(idx, *operand_ptr_, reduce_dim_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) shape[i] = Size(i);
    return shape;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Mean::Map(indexes, *operand_ptr_, reduce_dim_);
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");

    UnaryGradImpl<typename Operator::Mean::Grad, GIType, OIType> out_grad(
        grad, *operand_ptr_, reduce_dim_);
    operand_ptr_.InvokeBackward(out_grad);
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  Index reduce_dim_;
};

template <typename OIType>
class UnaryExpImpl<Operator::Max, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::Max, OIType>> {
 public:
  using Operator = Operator::Max;
  using operand_type = OIType;

  explicit UnaryExpImpl(const ExpImplPtr<OIType> &ptr, Index reduce_dim)
      : operand_ptr_(ptr, true), reduce_dim_(reduce_dim) {}

  Index DimensionsSize() const {
    return Operator::Max::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::Max::Size(idx, *operand_ptr_, reduce_dim_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Max::Map(indexes, *operand_ptr_, reduce_dim_);
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");

    UnaryGradImpl<typename Operator::Max::Grad, GIType, OIType> out_grad(
        grad, *operand_ptr_, reduce_dim_);
    operand_ptr_.InvokeBackward(out_grad);
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  Index reduce_dim_;
};

template <typename OIType>
class UnaryExpImpl<Operator::Argmax, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::Argmax, OIType>> {
 public:
  using Operator = Operator::Argmax;
  using operand_type = OIType;

  explicit UnaryExpImpl(const ExpImplPtr<OIType> &ptr, Index reduce_dim)
      : operand_ptr_(ptr, true), reduce_dim_(reduce_dim) {}

  Index DimensionsSize() const {
    return Operator::Argmax::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::Argmax::Size(idx, *operand_ptr_, reduce_dim_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  Index Eval(IndexArray &indexes) const {
    return Operator::Argmax::Map(indexes, *operand_ptr_, reduce_dim_);
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &) const {
    THROW_ERROR("Not implement error for backward of argmax.");
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  Index reduce_dim_;
};

template <typename OIType>
class UnaryExpImpl<Operator::NLLLoss, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::NLLLoss, OIType>> {
 public:
  using Operator = Operator::NLLLoss;
  using operand_type = OIType;

  explicit UnaryExpImpl(const ExpImplPtr<OIType> &ptr,
                        const std::shared_ptr<Index> &batch_label)
      : operand_ptr_(ptr, true), batch_label_(batch_label) {}

  Index DimensionsSize() const {
    return Operator::NLLLoss::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::NLLLoss::Size(idx, *operand_ptr_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::NLLLoss::Map(indexes, *operand_ptr_, batch_label_.get());
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");

    UnaryGradImpl<typename Operator::NLLLoss::Grad, GIType, OIType> out_grad(
        grad, *operand_ptr_, batch_label_.get());
    operand_ptr_.InvokeBackward(out_grad);
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  std::shared_ptr<Index> batch_label_;
};

template <typename OIType>
class UnaryExpImpl<Operator::Img2col, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::Img2col, OIType>> {
 public:
  using Operator = Operator::Img2col;
  using operand_type = OIType;

  UnaryExpImpl(const ExpImplPtr<OIType> &ptr,
               Operator::Img2col::MatrixSize kernel_size,
               Operator::Img2col::MatrixSize stride_size,
               Operator::Img2col::MatrixSize padding_size)
      : operand_ptr_(ptr, true),
        kernel_size_(std::move(kernel_size)),
        stride_size_(std::move(stride_size)),
        padding_size_(std::move(padding_size)) {
    Index b = operand_ptr_->Size(0);
    Index c = operand_ptr_->Size(1);
    Index h = operand_ptr_->Size(2);
    Index w = operand_ptr_->Size(3);
    out_size_.first = (h + 2 * padding_size_.first - kernel_size_.first) /
                          stride_size_.first +
                      1;
    out_size_.second = (w + 2 * padding_size_.second - kernel_size_.second) /
                           stride_size_.second +
                       1;
    shape_.first = out_size_.first * out_size_.second * b;
    shape_.second = c * kernel_size_.first * kernel_size_.second;
  }

  Index DimensionsSize() const {
    return Operator::Img2col::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::Img2col::Size(idx, *operand_ptr_, shape_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  const Operator::Img2col::MatrixSize &ConvFeatSize() const {
    return out_size_;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Img2col::Map(indexes, *operand_ptr_, kernel_size_,
                                  stride_size_, padding_size_, out_size_);
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &grad) {
    CHECK_EQUAL(this->GradCount(), 0, "Reused ExpImpl can't be Backward.");
    UnaryGradImpl<typename Operator::Img2col::Grad, GIType, OIType> out_grad(
        grad, *operand_ptr_, kernel_size_, stride_size_, padding_size_,
        out_size_);
    operand_ptr_.InvokeBackward(out_grad);
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  Operator::Img2col::MatrixSize kernel_size_;
  Operator::Img2col::MatrixSize stride_size_;
  Operator::Img2col::MatrixSize padding_size_;
  Operator::Img2col::MatrixSize out_size_;
  Operator::Img2col::MatrixSize shape_;
};

template <typename OIType>
class UnaryExpImpl<Operator::MaxPool2d, OIType>
    : public ExpImpl<UnaryExpImpl<Operator::MaxPool2d, OIType>> {
 public:
  using Operator = Operator::MaxPool2d;
  using operand_type = OIType;

  UnaryExpImpl(const ExpImplPtr<OIType> &ptr,
               Operator::MaxPool2d::MatrixSize kernel_size,
               Operator::MaxPool2d::MatrixSize stride_size,
               Operator::MaxPool2d::MatrixSize padding_size)
      : operand_ptr_(ptr, true),
        kernel_size_(std::move(kernel_size)),
        stride_size_(std::move(stride_size)),
        padding_size_(std::move(padding_size)) {
    Index h = operand_ptr_->Size(2);
    Index w = operand_ptr_->Size(3);
    out_size_.first = (h + 2 * padding_size_.first - kernel_size_.first) /
                          stride_size_.first +
                      1;
    out_size_.second = (w + 2 * padding_size_.second - kernel_size_.second) /
                           stride_size_.second +
                       1;
  }

  Index DimensionsSize() const {
    return Operator::MaxPool2d::DimensionsSize(*operand_ptr_);
  }

  Index Size(Index idx) const {
    return Operator::MaxPool2d::Size(idx, *operand_ptr_, out_size_);
  }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::MaxPool2d::Map(indexes, *operand_ptr_, kernel_size_,
                                    stride_size_, padding_size_);
  }

  bool RequiresGrad() const { return operand_ptr_->RequiresGrad(); }

  template <typename GIType>
  void Backward(const GIType &) {
    THROW_ERROR("Not implement error in backward of MaxPool2d.");
  }

 private:
  ExpImplPtr<OIType> operand_ptr_;
  Operator::MaxPool2d::MatrixSize kernel_size_;
  Operator::MaxPool2d::MatrixSize stride_size_;
  Operator::MaxPool2d::MatrixSize padding_size_;
  Operator::MaxPool2d::MatrixSize out_size_;
};

template <>
class UnaryExpImpl<Operator::Constant, BasicData>
    : public ExpImpl<UnaryExpImpl<Operator::Constant, BasicData>> {
 public:
  using Operator = Operator::Constant;
  using operand_type = BasicData;

  UnaryExpImpl(BasicData value, IndexArray &&Size)
      : value_(value), shape_(std::move(Size)) {}

  UnaryExpImpl(BasicData value, const IndexArray &Size)
      : value_(value), shape_(Size) {}

  Index DimensionsSize() const { return shape_.ArraySize(); }

  Index Size(Index idx) const { return shape_[idx]; }

  IndexArray Size() const {
    IndexArray shape(DimensionsSize());
    for (Index i = 0; i < shape.ArraySize(); ++i) {
      shape[i] = Size(i);
    }
    return shape;
  }

  BasicData Eval(IndexArray &indexes) const {
    return Operator::Constant::Map(indexes, value_);
  }

  bool RequiresGrad() const { return false; }

  template <typename GIType>
  void Backward(const GIType &grad) {
    THROW_ERROR("NotImplementError in Backward of Constant");
  }

 private:
  BasicData value_;
  IndexArray shape_;
};

}  // namespace KD
#endif