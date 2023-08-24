#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_MDARRAY_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_MDARRAY_H

#include <initializer_list>
#include <memory>

#include "exp/exp.h"
#include "exp/exp_impl.h"
#include "mdarray/mdarray_impl.h"

namespace KD {

template struct Exp<MdarrayImpl>;

class Mdarray : public Exp<MdarrayImpl> {
 public:
  Mdarray(const Storage &storage, const Shape &shape, const IndexArray &stride,
          bool requires_grad = false)
      : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(
            storage, shape, stride, requires_grad)) {}

  Mdarray(const Storage &storage, const Shape &shape,
          bool requires_grad = false)
      : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(
            storage, shape, requires_grad)) {}

  Mdarray(const BasicData *data, const Shape &shape, bool requires_grad = false)
      : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(
            data, shape, requires_grad)) {}

  explicit Mdarray(const Shape &shape, bool requires_grad = false)
      : Exp<MdarrayImpl>(
            Allocator::UniqueConstruct<MdarrayImpl>(shape, requires_grad)) {}

  explicit Mdarray(MdarrayImpl &&impl)
      : Exp<MdarrayImpl>(
            Allocator::UniqueConstruct<MdarrayImpl>(std::move(impl))) {}

  explicit Mdarray(Allocator::UniquePtr<MdarrayImpl> &&ptr)
      : Exp<MdarrayImpl>(std::move(ptr)) {}

  template <typename ImplType>
  Mdarray(const Exp<ImplType> &exp)
      : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(exp.Impl())) {}

  Mdarray(const Mdarray &other) = default;

  Mdarray(Mdarray &&other) = default;

  Mdarray &operator=(const Mdarray &other) {
    impl_ptr_->operator=(other.Impl());
    return *this;
  }

  Index DimensionsSize() const { return impl_ptr_->DimensionsSize(); }

  Index Size(Index idx) const { return impl_ptr_->Size(idx); }

  Index Offset() const { return impl_ptr_->Offset(); }

  const Shape &Size() const { return impl_ptr_->Size(); }

  const Storage &GetStorage() const { return impl_ptr_->GetStorage(); }

  const IndexArray &GetStride() const { return impl_ptr_->GetStride(); }

  Index Version() const { return impl_ptr_->Version(); }

  bool IsContiguous() const { return impl_ptr_->IsContiguous(); }

  BasicData &operator[](std::initializer_list<Index> ids) {
    return impl_ptr_->operator[](ids);
  }

  BasicData operator[](std::initializer_list<Index> ids) const {
    return impl_ptr_->operator[](ids);
  }

  BasicData Item() const { return impl_ptr_->Item(); }

  Mdarray Slice(Index dim, Index idx) const {
    return Mdarray{impl_ptr_->Slice(dim, idx)};
  }

  Mdarray Slice(Index dim, Index start_idx, Index end_idx) const {
    return Mdarray{impl_ptr_->Slice(dim, start_idx, end_idx)};
  }

  Mdarray Transpose(Index dim_1, Index dim_2) const {
    return Mdarray{impl_ptr_->Transpose(dim_1, dim_2)};
  }

  Mdarray Permute(std::initializer_list<Index> dims) const {
    return Mdarray{impl_ptr_->Permute(dims)};
  }

  Mdarray View(const Shape &shape) const {
    return Mdarray{impl_ptr_->View(shape)};
  }

  Mdarray Squeeze() const { return Mdarray{impl_ptr_->Squeeze()}; }

  Mdarray Unsqueeze(Index dim) const {
    return Mdarray{impl_ptr_->Unsqueeze(dim)};
  }

  template <typename ImplType>
  Mdarray &operator=(const Exp<ImplType> &exp) {
    impl_ptr_->operator=(exp.Impl());
    return *this;
  }

  template <typename ImplType>
  Mdarray &operator+=(const Exp<ImplType> &exp) {
    impl_ptr_->operator+=(exp.Impl());
    return *this;
  }

  Mdarray Grad() const { return Mdarray{impl_ptr_->Grad()}; }

  void Backward() {
    CHECK_TRUE(
        impl_ptr_->RequiresGrad(),
        "Multidimensional arrays doesn't require grad and doesn't have a "
        "grad_fn.")
    impl_ptr_.InvokeBackward(UnaryGradImpl<Operator::Constant, void, BasicData>(
        1, static_cast<const IndexArray &>(this->Size())));
  }

  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const Mdarray &t) {
    stream << *(t.impl_ptr_);
    stream << std::endl;
    stream << "Shape: " << t.Size();
    return stream;
  }
};

}  // namespace KD
#endif
