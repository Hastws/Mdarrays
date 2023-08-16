#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MULTIDIMENSIONAL_ARRAYS_MULTIDIMENSIONAL_ARRAYS_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MULTIDIMENSIONAL_ARRAYS_MULTIDIMENSIONAL_ARRAYS_H

#include <initializer_list>
#include <memory>

#include "exp/exp.h"
#include "exp/exp_impl.h"
#include "multidimensional_arrays/multidimensional_arrays_impl.h"

namespace KD {

template struct Exp<MultidimensionalArraysImpl>;

class MultidimensionalArrays : public Exp<MultidimensionalArraysImpl> {
 public:
  MultidimensionalArrays(const Storage &storage, const Shape &shape,
                         const IndexArray &stride, bool requires_grad = false);

  MultidimensionalArrays(const Storage &storage, const Shape &shape,
                         bool requires_grad = false);

  MultidimensionalArrays(const BasicData *data, const Shape &shape,
                         bool requires_grad = false);

  explicit MultidimensionalArrays(const Shape &shape,
                                  bool requires_grad = false);

  explicit MultidimensionalArrays(MultidimensionalArraysImpl &&impl);

  explicit MultidimensionalArrays(
      Allocator::UniquePtr<MultidimensionalArraysImpl> &&ptr);

  template <typename ImplType>
  MultidimensionalArrays(const Exp<ImplType> &exp);

  MultidimensionalArrays(const MultidimensionalArrays &other) = default;

  MultidimensionalArrays(MultidimensionalArrays &&other) = default;

  MultidimensionalArrays &operator=(const MultidimensionalArrays &other);

  Index DimensionsSize() const;

  Index Size(Index idx) const;

  const Shape &Size() const;

  Index Offset() const;

  const IndexArray &Stride() const;

  Index Version() const;

  bool IsContiguous() const;

  MultidimensionalArrays Grad() const;

  BasicData &operator[](std::initializer_list<Index> ids);

  BasicData operator[](std::initializer_list<Index> ids) const;

  BasicData Item() const;

  MultidimensionalArrays Slice(Index dim, Index idx) const;

  MultidimensionalArrays Slice(Index dim, Index start_idx, Index end_idx) const;

  MultidimensionalArrays Transpose(Index dim1, Index dim2) const;

  MultidimensionalArrays View(const Shape &shape) const;

  MultidimensionalArrays Squeeze() const;

  MultidimensionalArrays Unsqueeze(Index dim) const;

  MultidimensionalArrays Permute(std::initializer_list<Index> dims) const;

  template <typename ImplType>
  MultidimensionalArrays &operator=(const Exp<ImplType> &exp);

  template <typename ImplType>
  MultidimensionalArrays &operator+=(const Exp<ImplType> &exp);

  void Backward();

  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const MultidimensionalArrays &t);
};

template <typename Stream>
Stream &operator<<(Stream &stream, const MultidimensionalArrays &t) {
  stream << *(t.impl_ptr_);
  stream << std::endl;
  stream << "Shape: " << t.Size();
  return stream;
}

template <typename ImplType>
MultidimensionalArrays::MultidimensionalArrays(const Exp<ImplType> &exp)
    : Exp<MultidimensionalArraysImpl>(
          Allocator::UniqueConstruct<MultidimensionalArraysImpl>(exp.Impl())) {}

template <typename ImplType>
MultidimensionalArrays &MultidimensionalArrays::operator=(
    const Exp<ImplType> &exp) {
  impl_ptr_->operator=(exp.Impl());
  return *this;
}

template <typename ImplType>
MultidimensionalArrays &MultidimensionalArrays::operator+=(
    const Exp<ImplType> &exp) {
  impl_ptr_->operator+=(exp.Impl());
  return *this;
}

}  // namespace KD
#endif
