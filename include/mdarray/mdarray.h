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
          bool requires_grad = false);

  Mdarray(const Storage &storage, const Shape &shape,
          bool requires_grad = false);

  Mdarray(const BasicData *data, const Shape &shape,
          bool requires_grad = false);

  explicit Mdarray(const Shape &shape, bool requires_grad = false);

  explicit Mdarray(MdarrayImpl &&impl);

  explicit Mdarray(Allocator::UniquePtr<MdarrayImpl> &&ptr);

  template <typename ImplType>
  Mdarray(const Exp<ImplType> &exp);

  Mdarray(const Mdarray &other) = default;

  Mdarray(Mdarray &&other) = default;

  Mdarray &operator=(const Mdarray &other);

  Index DimensionsSize() const;

  Index Size(Index idx) const;

  const Shape &Size() const;

  Index Offset() const;

  const IndexArray &Stride() const;

  Index Version() const;

  bool IsContiguous() const;

  Mdarray Grad() const;

  BasicData &operator[](std::initializer_list<Index> ids);

  BasicData operator[](std::initializer_list<Index> ids) const;

  BasicData Item() const;

  Mdarray Slice(Index dim, Index idx) const;

  Mdarray Slice(Index dim, Index start_idx, Index end_idx) const;

  Mdarray Transpose(Index dim1, Index dim2) const;

  Mdarray View(const Shape &shape) const;

  Mdarray Squeeze() const;

  Mdarray Unsqueeze(Index dim) const;

  Mdarray Permute(std::initializer_list<Index> dims) const;

  template <typename ImplType>
  Mdarray &operator=(const Exp<ImplType> &exp);

  template <typename ImplType>
  Mdarray &operator+=(const Exp<ImplType> &exp);

  void Backward();

  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const Mdarray &t);
};

template <typename Stream>
Stream &operator<<(Stream &stream, const Mdarray &t) {
  stream << *(t.impl_ptr_);
  stream << std::endl;
  stream << "Shape: " << t.Size();
  return stream;
}

template <typename ImplType>
Mdarray::Mdarray(const Exp<ImplType> &exp)
    : Exp<MdarrayImpl>(Allocator::UniqueConstruct<MdarrayImpl>(exp.Impl())) {}

template <typename ImplType>
Mdarray &Mdarray::operator=(const Exp<ImplType> &exp) {
  impl_ptr_->operator=(exp.Impl());
  return *this;
}

template <typename ImplType>
Mdarray &Mdarray::operator+=(const Exp<ImplType> &exp) {
  impl_ptr_->operator+=(exp.Impl());
  return *this;
}

}  // namespace KD
#endif
