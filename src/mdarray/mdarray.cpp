#include "mdarray/mdarray.h"

#include "exp/grad_impl.h"
#include "exp/operator/constant.h"
#include "mdarray/shape.h"

namespace KD {

Mdarray::Mdarray(const Storage &storage, const Shape &shape,
                 const IndexArray &stride, bool requires_grad)
    : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(
          storage, shape, stride, requires_grad)) {}

Mdarray::Mdarray(const Storage &storage, const Shape &shape, bool requires_grad)
    : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(
          storage, shape, requires_grad)) {}

Mdarray::Mdarray(const BasicData *data, const Shape &shape, bool requires_grad)
    : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(
          data, shape, requires_grad)) {}

Mdarray::Mdarray(const Shape &shape, bool requires_grad)
    : Exp<MdarrayImpl>(
          Allocator::UniqueConstruct<MdarrayImpl>(shape, requires_grad)) {}

Mdarray::Mdarray(MdarrayImpl &&impl)
    : Exp<MdarrayImpl>(
          Allocator::UniqueConstruct<MdarrayImpl>(std::move(impl))) {}

Mdarray::Mdarray(Allocator::UniquePtr<MdarrayImpl> &&ptr)
    : Exp<MdarrayImpl>(std::move(ptr)) {}

Mdarray &Mdarray::operator=(const Mdarray &other) {
  impl_ptr_->operator=(other.Impl());
  return *this;
}

Index Mdarray::DimensionsSize() const { return impl_ptr_->DimensionsSize(); }

Index Mdarray::Size(Index idx) const { return impl_ptr_->Size(idx); }

const Shape &Mdarray::Size() const { return impl_ptr_->Size(); }

Index Mdarray::Offset() const { return impl_ptr_->Offset(); }

const IndexArray &Mdarray::Stride() const { return impl_ptr_->Stride(); }

Index Mdarray::Version() const { return impl_ptr_->Version(); }

bool Mdarray::IsContiguous() const { return impl_ptr_->IsContiguous(); }

BasicData &Mdarray::operator[](std::initializer_list<Index> ids) {
  return impl_ptr_->operator[](ids);
}

BasicData Mdarray::operator[](std::initializer_list<Index> ids) const {
  return impl_ptr_->operator[](ids);
}

BasicData Mdarray::Item() const { return impl_ptr_->Item(); }

Mdarray Mdarray::Slice(Index dim, Index idx) const {
  return Mdarray{impl_ptr_->Slice(dim, idx)};
}

Mdarray Mdarray::Slice(Index dim, Index start_idx, Index end_idx) const {
  return Mdarray{impl_ptr_->Slice(dim, start_idx, end_idx)};
}

Mdarray Mdarray::Transpose(Index dim1, Index dim2) const {
  return Mdarray{impl_ptr_->Transpose(dim1, dim2)};
}

Mdarray Mdarray::Permute(std::initializer_list<Index> dims) const {
  return Mdarray{impl_ptr_->Permute(dims)};
}

Mdarray Mdarray::View(const Shape &shape) const {
  return Mdarray{impl_ptr_->View(shape)};
}

Mdarray Mdarray::Squeeze() const { return Mdarray{impl_ptr_->Squeeze()}; }

Mdarray Mdarray::Unsqueeze(Index dim) const {
  return Mdarray{impl_ptr_->Unsqueeze(dim)};
}

Mdarray Mdarray::Grad() const { return Mdarray{impl_ptr_->Grad()}; }

void Mdarray::Backward() {
  CHECK_TRUE(impl_ptr_->RequiresGrad(),
             "Multidimensional arrays doesn't require grad and doesn't have a "
             "grad_fn.");
  impl_ptr_.InvokeBackward(UnaryGradImpl<Operator::Constant, void, BasicData>(
      1, static_cast<const IndexArray &>(this->Size())));
}

}  // namespace KD
