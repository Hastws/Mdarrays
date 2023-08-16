#include "multidimensional_arrays/multidimensional_arrays.h"

#include "exp/grad_impl.h"
#include "exp/operator/constant.h"
#include "multidimensional_arrays/shape.h"

namespace KD {

MultidimensionalArrays::MultidimensionalArrays(const Storage &storage,
                                               const Shape &shape,
                                               const IndexArray &stride,
                                               bool requires_grad)
    : Exp<MultidimensionalArraysImpl>(
          Allocator::UniqueConstruct<MultidimensionalArraysImpl>(
              storage, shape, stride, requires_grad)) {}

MultidimensionalArrays::MultidimensionalArrays(const Storage &storage,
                                               const Shape &shape,
                                               bool requires_grad)
    : Exp<MultidimensionalArraysImpl>(
          Allocator::UniqueConstruct<MultidimensionalArraysImpl>(
              storage, shape, requires_grad)) {}

MultidimensionalArrays::MultidimensionalArrays(const BasicData *data,
                                               const Shape &shape,
                                               bool requires_grad)
    : Exp<MultidimensionalArraysImpl>(
          Allocator::UniqueConstruct<MultidimensionalArraysImpl>(
              data, shape, requires_grad)) {}

MultidimensionalArrays::MultidimensionalArrays(const Shape &shape,
                                               bool requires_grad)
    : Exp<MultidimensionalArraysImpl>(
          Allocator::UniqueConstruct<MultidimensionalArraysImpl>(
              shape, requires_grad)) {}

MultidimensionalArrays::MultidimensionalArrays(
    MultidimensionalArraysImpl &&impl)
    : Exp<MultidimensionalArraysImpl>(
          Allocator::UniqueConstruct<MultidimensionalArraysImpl>(
              std::move(impl))) {}

MultidimensionalArrays::MultidimensionalArrays(
    Allocator::UniquePtr<MultidimensionalArraysImpl> &&ptr)
    : Exp<MultidimensionalArraysImpl>(std::move(ptr)) {}

MultidimensionalArrays &MultidimensionalArrays::operator=(
    const MultidimensionalArrays &other) {
  impl_ptr_->operator=(other.Impl());
  return *this;
}

Index MultidimensionalArrays::DimensionsSize() const {
  return impl_ptr_->DimensionsSize();
}

Index MultidimensionalArrays::Size(Index idx) const {
  return impl_ptr_->Size(idx);
}

const Shape &MultidimensionalArrays::Size() const { return impl_ptr_->Size(); }

Index MultidimensionalArrays::Offset() const { return impl_ptr_->Offset(); }

const IndexArray &MultidimensionalArrays::Stride() const {
  return impl_ptr_->Stride();
}

Index MultidimensionalArrays::Version() const { return impl_ptr_->Version(); }

bool MultidimensionalArrays::IsContiguous() const {
  return impl_ptr_->IsContiguous();
}

BasicData &MultidimensionalArrays::operator[](
    std::initializer_list<Index> ids) {
  return impl_ptr_->operator[](ids);
}

BasicData MultidimensionalArrays::operator[](
    std::initializer_list<Index> ids) const {
  return impl_ptr_->operator[](ids);
}

BasicData MultidimensionalArrays::Item() const { return impl_ptr_->Item(); }

MultidimensionalArrays MultidimensionalArrays::Slice(Index dim,
                                                     Index idx) const {
  return MultidimensionalArrays{impl_ptr_->Slice(dim, idx)};
}

MultidimensionalArrays MultidimensionalArrays::Slice(Index dim, Index start_idx,
                                                     Index end_idx) const {
  return MultidimensionalArrays{impl_ptr_->Slice(dim, start_idx, end_idx)};
}

MultidimensionalArrays MultidimensionalArrays::Transpose(Index dim1,
                                                         Index dim2) const {
  return MultidimensionalArrays{impl_ptr_->Transpose(dim1, dim2)};
}

MultidimensionalArrays MultidimensionalArrays::Permute(
    std::initializer_list<Index> dims) const {
  return MultidimensionalArrays{impl_ptr_->Permute(dims)};
}

MultidimensionalArrays MultidimensionalArrays::View(const Shape &shape) const {
  return MultidimensionalArrays{impl_ptr_->View(shape)};
}

MultidimensionalArrays MultidimensionalArrays::Squeeze() const {
  return MultidimensionalArrays{impl_ptr_->Squeeze()};
}

MultidimensionalArrays MultidimensionalArrays::Unsqueeze(Index dim) const {
  return MultidimensionalArrays{impl_ptr_->Unsqueeze(dim)};
}

MultidimensionalArrays MultidimensionalArrays::Grad() const {
  return MultidimensionalArrays{impl_ptr_->Grad()};
}

void MultidimensionalArrays::Backward() {
  CHECK_TRUE(impl_ptr_->RequiresGrad(),
             "Multidimensional arrays doesn't require grad and doesn't have a "
             "grad_fn.");
  impl_ptr_.InvokeBackward(UnaryGradImpl<Operator::Constant, void, BasicData>(
      1, static_cast<const IndexArray &>(this->Size())));
}

}  // namespace KD
