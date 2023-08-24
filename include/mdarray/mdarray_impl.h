#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_MDARRAY_IMPL_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_MDARRAY_IMPL_H

#include <initializer_list>
#include <utility>

#include "exp/exp.h"
#include "exp/exp_impl.h"
#include "exp/operator/basic_op.h"
#include "mdarray/shape.h"
#include "mdarray/storage.h"
#include "utils/exception.h"

namespace KD {

// forward declaration
struct AutoGradMeta;

class MdarrayImplUniversalAgent;

// Multidimensional array
class MdarrayImpl : public ExpImpl<MdarrayImpl> {
 public:
  // To be consistent with UnaryImpl
  using Operator = Operator::Identity;
  using operand_type = MdarrayImpl;

  // constructor
  MdarrayImpl(const Storage &storage, const Shape &shape,
              const IndexArray &stride, bool requires_grad = false);

  MdarrayImpl(Storage storage, Shape shape, bool requires_grad = false);

  MdarrayImpl(const BasicData *data, const Shape &shape,
              bool requires_grad = false);

  explicit MdarrayImpl(const Shape &shape, bool requires_grad = false);

  MdarrayImpl(Storage &&storage, Shape &&shape, IndexArray &&stride,
              bool requires_grad = false);

  template <typename ImplType>
  MdarrayImpl(const ImplType &impl);

  MdarrayImpl(const MdarrayImpl &other) = delete;

  MdarrayImpl(MdarrayImpl &&other) = default;

  MdarrayImpl &operator=(const MdarrayImpl &other);

  // inline function
  Index DimensionsSize() const { return shape_.DimensionsSize(); }

  Index Size(Index idx) const { return shape_[idx]; }

  const Shape &Size() const { return shape_; }

  const IndexArray &GetStride() const { return stride_; }

  const Storage &GetStorage() const { return storage_; }

  Index Offset() const { return storage_.Offset(); }

  Index Version() const { return storage_.Version(); }

  bool RequiresGrad() const { return requires_grad_; }

  // other method
  bool IsContiguous() const;

  Allocator::UniquePtr<MdarrayImpl> Grad() const;

  BasicData &operator[](std::initializer_list<Index> indexes);

  BasicData operator[](std::initializer_list<Index> indexes) const;

  BasicData Item() const;

  Allocator::UniquePtr<MdarrayImpl> Slice(Index dim, Index idx) const;

  Allocator::UniquePtr<MdarrayImpl> Slice(Index dim, Index start_idx,
                                          Index end_idx) const;

  Allocator::UniquePtr<MdarrayImpl> Transpose(Index dim1, Index dim2) const;

  Allocator::UniquePtr<MdarrayImpl> View(const Shape &shape) const;

  Allocator::UniquePtr<MdarrayImpl> Squeeze() const;

  Allocator::UniquePtr<MdarrayImpl> Unsqueeze(Index dim) const;

  Allocator::UniquePtr<MdarrayImpl> Permute(
      std::initializer_list<Index> dims) const;

  // member function for expression template
  BasicData Eval(IndexArray &indexes) const {
    Index offset = 0;
    for (KD::Index i = 0; i < DimensionsSize(); ++i) {
      offset += indexes[i] * stride_[i];
    }
    return storage_[offset];
  }

  template <typename ImplType>
  MdarrayImpl &operator=(const ImplType &exp_impl);

  template <typename ImplType>
  MdarrayImpl &operator+=(const ImplType &exp_impl);

  friend ExpImplPtr<MdarrayImpl>;

  friend class MdarrayImplUniversalAgent;

 private:
  template <typename ImplType>
  void Backward(const ImplType &grad);

  void Backward();

  Storage storage_;
  Shape shape_;
  IndexArray stride_;

  bool requires_grad_;
  Allocator::UniquePtr<AutoGradMeta> grad_meta_ptr_;
};

class MdarrayImplUniversalAgent {
 public:
  explicit MdarrayImplUniversalAgent(const MdarrayImpl &mdarray_impl)
      : mdarray_impl_(mdarray_impl) {}
  AutoGradMeta *GetGradMetaPtr() { return mdarray_impl_.grad_meta_ptr_.get(); }

 private:
  const MdarrayImpl &mdarray_impl_;
};

template <typename Stream>
Stream &operator<<(Stream &stream, const MdarrayImpl &src) {
  MdarrayImpl t(src.Size());
  t = src;

  stream << '[';
  if (t.DimensionsSize() == 1) {
    stream << t[{0}];
    for (Index i = 1; i < t.Size(0); i++) {
      stream << ", " << t[{i}];
    }
  } else if (t.DimensionsSize() == 2) {
    stream << *t.Slice(0, 0);
    for (Index i = 1; i < t.Size(0); i++) {
      stream << ',';
      stream << *t.Slice(0, i);
    }
  } else {
    stream << *t.Slice(0, 0);
    for (Index i = 1; i < t.Size(0); i++) {
      stream << ',';
      stream << *t.Slice(0, i);
    }
  }
  stream << ']';

  return stream;
}

// Template specialization for ExpImplPtr
template <>
class ExpImplPtr<MdarrayImpl> {
 public:
  ExpImplPtr(Allocator::UniquePtr<MdarrayImpl> &&ptr, bool with_grad)
      : ptr_(ptr.release()),
        with_grad_(with_grad &&
                   static_cast<MdarrayImpl *>(ptr_)->RequiresGrad()),
        version_(static_cast<MdarrayImpl *>(ptr_)->Version()) {
    IncrementCounters();
  }

  ExpImplPtr(const MdarrayImpl &impl, bool with_grad)
      : ptr_(const_cast<MdarrayImpl *>(&impl)),
        with_grad_(with_grad &&
                   static_cast<MdarrayImpl *>(ptr_)->RequiresGrad()),
        version_(static_cast<MdarrayImpl *>(ptr_)->Version()) {
    IncrementCounters();
  }

  ExpImplPtr(const ExpImplPtr &other, bool with_grad)
      : ptr_(other.ptr_),
        with_grad_(with_grad &&
                   static_cast<MdarrayImpl *>(ptr_)->RequiresGrad()),
        version_(static_cast<MdarrayImpl *>(ptr_)->Version()) {
    IncrementCounters();
  }

  ~ExpImplPtr() { DecreaseRefcount(); }

  MdarrayImpl *operator->() const { return static_cast<MdarrayImpl *>(ptr_); }

  const MdarrayImpl &operator*() const {
    return *static_cast<MdarrayImpl *>(ptr_);
  }

  explicit operator bool() const { return ptr_ != nullptr; }

  template <typename GradImplType>
  void InvokeBackward(const GradImplType &grad) {
    MdarrayImpl *ptr = static_cast<MdarrayImpl *>(ptr_);
    if (ptr->RequiresGrad()) {
      CHECK_EQUAL(version_, ptr->Version(),
                  "Leaf variable has been moved into the graph interior");
      if (with_grad_) {
        ptr->DecreaseGradCount();
      }
      ptr->Backward(grad);
    }
  }

  void InvokeBackward() {
    MdarrayImpl *ptr = static_cast<MdarrayImpl *>(ptr_);
    if (ptr->RequiresGrad()) {
      CHECK_EQUAL(version_, ptr->Version(),
                  "Leaf variable has been moved into the graph interior");
      if (with_grad_) {
        ptr->DecreaseGradCount();
      }
      ptr->Backward();
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
      delete_handler_(static_cast<void *>(ptr_));
    }
  }

  ExpImpl<MdarrayImpl> *ptr_;
  bool with_grad_;
  Index version_;
  Allocator::DeleteHandler<MdarrayImpl> delete_handler_;
};
}  // namespace KD

namespace KD {
struct GradFn {
  virtual void operator()() = 0;

  virtual void operator()(const Storage &grad, const Shape &shape,
                          const IndexArray &stride) = 0;

  virtual ~GradFn() = default;

  struct MdarrayGradImpl : public GradImpl<MdarrayGradImpl> {
    const Storage &storage_;
    const Shape &shape_;
    const IndexArray &stride_;

    MdarrayGradImpl(const Storage &storage, const Shape &shape,
                    const IndexArray &stride)
        : storage_(storage), shape_(shape), stride_(stride) {}

    BasicData Eval(IndexArray &indexes) const {
      Index offset = 0;
      for (KD::Index i = 0; i < shape_.DimensionsSize(); ++i) {
        offset += indexes[i] * stride_[i];
      }
      return storage_[offset];
    }

    IndexArray GradSize() const {
      return static_cast<const IndexArray &>(shape_);
    }
  };
};

template <typename ImplType>
class GradFnImpl : public GradFn {
 public:
  GradFnImpl(const ImplType &impl) : next_exp_(impl, false) {}

  ~GradFnImpl() override = default;

  void operator()() override {
    THROW_ERROR("Need Grad when invoke Backward method of a expression.");
  }

  void operator()(const Storage &grad, const Shape &shape,
                  const IndexArray &stride) override {
    MdarrayGradImpl grad_exp_impl(grad, shape, stride);
    next_exp_.InvokeBackward(grad_exp_impl);
  }

 private:
  ExpImplPtr<ImplType> next_exp_;
};

template <>
struct GradFnImpl<MdarrayImpl> : public GradFn {
 public:
  GradFnImpl(const MdarrayImpl &impl) : next_exp_(impl, false) {}

  ~GradFnImpl() override = default;

  void operator()() override { next_exp_.InvokeBackward(); }

  void operator()(const Storage &grad, const Shape &shape,
                  const IndexArray &stride) override {
    MdarrayGradImpl grad_exp_impl(grad, shape, stride);
    next_exp_.InvokeBackward(grad_exp_impl);
  }

 private:
  ExpImplPtr<MdarrayImpl> next_exp_;
};

struct AutoGradMeta {
  Storage grad_;
  bool from_view_;
  std::shared_ptr<GradFn> grad_fn_ptr_;

  AutoGradMeta(const Shape &multidimensional_arrays_shape)
      : grad_(multidimensional_arrays_shape.SpaceSize(), 0),
        from_view_(false),
        grad_fn_ptr_(nullptr) {}

  AutoGradMeta(const Storage &grad, Index offset)
      : grad_(grad, offset), from_view_(false), grad_fn_ptr_(nullptr) {}

  void SetFromView(bool from_view) { from_view_ = from_view; }

  template <typename ImplType>
  void SetGradFn(const ImplType &impl) {
    auto ptr = Allocator::SharedConstruct<GradFnImpl<ImplType>>(impl);
    grad_fn_ptr_ = ptr;
  }
};

template <typename ImplType>
void Assign(Storage &dist_storage, const Shape &dist_shape,
            const IndexArray &dist_stride, const ImplType &src_exp);

template <typename ImplType>
void InplacementAdd(Storage &dist_storage, const Shape &dist_shape,
                    const IndexArray &dist_stride, const ImplType &src_exp);

template <typename ImplType>
void AssignUncontiguous(Storage &dist_storage, const Shape &dist_shape,
                        const IndexArray &dist_stride, const ImplType &src_exp);

template <typename ImplType>
void InplacementAddUncontiguous(Storage &dist_storage, const Shape &dist_shape,
                                const IndexArray &dist_stride,
                                const ImplType &src_exp);

// member template function definition
template <typename ImplType>
MdarrayImpl::MdarrayImpl(const ImplType &impl)
    : MdarrayImpl(impl.Size(), impl.RequiresGrad()) {
  this->operator=(impl);
}

template <typename ImplType>
MdarrayImpl &MdarrayImpl::operator=(const ImplType &exp_impl) {
  CHECK_EXP_SAME_SHAPE(*this, exp_impl);

  if (requires_grad_) {
    grad_meta_ptr_->SetGradFn(exp_impl);
    grad_meta_ptr_->SetFromView(false);
    storage_.IncrementVersion();
  }

  if (IsContiguous())
    Assign(storage_, shape_, stride_, exp_impl);
  else
    AssignUncontiguous(storage_, shape_, stride_, exp_impl);
  return *this;
}

template <typename ImplType>
MdarrayImpl &MdarrayImpl::operator+=(const ImplType &exp_impl) {
  CHECK_EXP_SAME_SHAPE(*this, exp_impl);

  if (requires_grad_) {
    grad_meta_ptr_->SetGradFn(exp_impl);
    grad_meta_ptr_->SetFromView(false);
    storage_.IncrementVersion();
  }

  if (IsContiguous())
    InplacementAdd(storage_, shape_, stride_, exp_impl);
  else
    InplacementAddUncontiguous(storage_, shape_, stride_, exp_impl);
  return *this;
}

inline MdarrayImpl &MdarrayImpl::operator=(const MdarrayImpl &other) {
  CHECK_EXP_SAME_SHAPE(*this, other);
  if (requires_grad_) {
    grad_meta_ptr_->SetGradFn(other);
    grad_meta_ptr_->SetFromView(false);
    storage_.IncrementVersion();
  }

  if (IsContiguous())
    Assign(storage_, shape_, stride_, other);
  else
    AssignUncontiguous(storage_, shape_, stride_, other);
  return *this;
}

template <typename ImplType>
void MdarrayImpl::Backward(const ImplType &grad) {
  // If the gradient is from a non-broadcasting operation,
  // shape will be the same to this->shape_;
  // Otherwise, shape will be broadcast.
  Shape shape(grad.GradSize());
  if (IsContiguous() && shape == shape_) {
    InplacementAdd(grad_meta_ptr_->grad_, shape, stride_, grad);
  } else {
    InplacementAddUncontiguous(grad_meta_ptr_->grad_, shape, stride_, grad);
  }
  Backward();
}

inline void MdarrayImpl::Backward() {
  if (bool(grad_meta_ptr_->grad_fn_ptr_) && GradCount() == 0) {
    auto &grad_fn = *(grad_meta_ptr_->grad_fn_ptr_);
    if (grad_meta_ptr_->from_view_) {
      grad_fn();
    } else {
      grad_fn(grad_meta_ptr_->grad_, shape_, stride_);
    }
  }
}

template <typename ImplType>
void Assign(Storage &dist_storage, const Shape &dist_shape,
            const IndexArray &dist_stride, const ImplType &src_exp) {
  IndexArray indexes(dist_shape.DimensionsSize());
  for (Index i = 0; i < dist_shape.SpaceSize(); ++i) {
    for (Index ii = i, j = 0; j < dist_shape.DimensionsSize(); ++j) {
      if (dist_stride[j] != 0) {
        indexes[j] = ii / dist_stride[j];
        ii %= dist_stride[j];
      } else {
        indexes[j] = 0;
      }
    }
    dist_storage[i] = src_exp.Eval(indexes);
  }
}

template <typename ImplType>
void InplacementAdd(Storage &dist_storage, const Shape &dist_shape,
                    const IndexArray &dist_stride, const ImplType &src_exp) {
  IndexArray indexes(dist_shape.DimensionsSize());
  for (Index i = 0; i < dist_shape.SpaceSize(); ++i) {
    for (Index ii = i, j = 0; j < dist_shape.DimensionsSize(); ++j) {
      if (dist_stride[j] != 0) {
        indexes[j] = ii / dist_stride[j];
        ii %= dist_stride[j];
      } else {
        indexes[j] = 0;
      }
    }
    dist_storage[i] += src_exp.Eval(indexes);
  }
}

template <typename ImplType>
void AssignUncontiguous(Storage &dist_storage, const Shape &dist_shape,
                        const IndexArray &dist_stride,
                        const ImplType &src_exp) {
  IndexArray indexes(dist_shape.DimensionsSize());
  IndexArray cur(dist_shape.DimensionsSize());
  Index idx = 0;
  cur.Memset(0);

  while (true) {
    if (idx == dist_shape.DimensionsSize()) {
      --idx;
      Index offset = 0;
      for (Index i = 0; i < indexes.ArraySize(); ++i)
        offset += dist_stride[i] * indexes[i];
      dist_storage[offset] = src_exp.Eval(indexes);
    } else {
      while (idx < dist_shape.DimensionsSize() && cur[idx] == dist_shape[idx]) {
        cur[idx] = 0;
        --idx;
      }
      if (idx > dist_shape.DimensionsSize()) {
        break;
      }

      indexes[idx] = cur[idx];
      ++cur[idx];
      ++idx;
    }
  }
}

template <typename ImplType>
void InplacementAddUncontiguous(Storage &dist_storage, const Shape &dist_shape,
                                const IndexArray &dist_stride,
                                const ImplType &src_exp) {
  IndexArray indexes(dist_shape.DimensionsSize());
  IndexArray cur(dist_shape.DimensionsSize());
  Index idx = 0;
  cur.Memset(0);

  while (true) {
    if (idx == dist_shape.DimensionsSize()) {
      --idx;
      Index offset = 0;
      for (Index i = 0; i < indexes.ArraySize(); ++i)
        offset += dist_stride[i] * indexes[i];
      dist_storage[offset] += src_exp.Eval(indexes);
    } else {
      while (idx < dist_shape.DimensionsSize() && cur[idx] == dist_shape[idx]) {
        cur[idx] = 0;
        --idx;
      }
      if (idx > dist_shape.DimensionsSize()) break;

      indexes[idx] = cur[idx];
      ++cur[idx];
      ++idx;
    }
  }
}
}  // namespace KD
#endif