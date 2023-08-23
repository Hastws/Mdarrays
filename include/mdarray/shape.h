#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_SHAPE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_SHAPE_H

#include <initializer_list>
#include <ostream>

#include "memory_pool/allocator.h"
#include "utils/base_config.h"
#include "utils/fixed_array.h"

namespace KD {

class Shape {
 public:
  // constructor
  Shape(std::initializer_list<Index> dims) : dims_(dims) {}

  Shape(const Shape &other, Index skip) : dims_(other.DimensionsSize() - 1) {
    Index i = 0;
    for (; i < skip; ++i) dims_[i] = other.dims_[i];
    for (; i < dims_.ArraySize(); ++i) dims_[i] = other.dims_[i + 1];
  }

  Shape(Index *dims, Index dim_) : dims_(dims, dim_) {}

  Shape(IndexArray &&shape) : dims_(std::move(shape)) {}

  Shape(const Shape &other) = default;

  Shape(Shape &&other) = default;

  ~Shape() = default;

  // method
  Index SpaceSize() const {
    Index res = 1;
    for (Index i = 0; i < dims_.ArraySize(); ++i) res *= dims_[i];
    return res;
  }

  Index SubSpaceSize(Index start_dim, Index end_dim) const {
    Index res = 1;
    for (; start_dim < end_dim; ++start_dim) res *= dims_[start_dim];
    return res;
  }

  Index SubSpaceSize(Index start_dim) const {
    return SubSpaceSize(start_dim, dims_.ArraySize());
  }

  bool operator==(const Shape &other) const {
    if (this->DimensionsSize() != other.DimensionsSize()) return false;
    Index i = 0;
    for (; i < dims_.ArraySize() && dims_[i] == other.dims_[i]; ++i)
      ;
    return i == dims_.ArraySize();
  }

  // inline function
  Index DimensionsSize() const { return dims_.ArraySize(); }

  Index operator[](Index idx) const { return dims_[idx]; }

  Index &operator[](Index idx) { return dims_[idx]; }

  explicit operator const IndexArray &() const { return dims_; }

 private:
  IndexArray dims_;
};

template <typename Stream>
Stream &operator<<(Stream &stream, const Shape &s) {
  stream << '(' << s[0];
  for (KD::Index i = 1; i < s.DimensionsSize(); ++i) {
    stream << ", " << s[i];
  }
  stream << ")";
  return stream;
}

}  // namespace KD

#endif