#include "multidimensional_arrays/shape.h"

namespace KD {

Shape::Shape(std::initializer_list<Index> dims) : dims_(dims) {}

Shape::Shape(const Shape &other, Index skip)
    : dims_(other.DimensionsSize() - 1) {
  Index i = 0;
  for (; i < skip; ++i) dims_[i] = other.dims_[i];
  for (; i < dims_.ArraySize(); ++i) dims_[i] = other.dims_[i + 1];
}

Shape::Shape(Index *dims, Index dim_) : dims_(dims, dim_) {}

Shape::Shape(IndexArray &&shape) : dims_(std::move(shape)) {}

Index Shape::SpaceSize() const {
  Index res = 1;
  for (Index i = 0; i < dims_.ArraySize(); ++i) res *= dims_[i];
  return res;
}

Index Shape::SubSpaceSize(Index start_dim, Index end_dim) const {
  Index res = 1;
  for (; start_dim < end_dim; ++start_dim) res *= dims_[start_dim];
  return res;
}

Index Shape::SubSpaceSize(Index start_dim) const {
  return SubSpaceSize(start_dim, dims_.ArraySize());
}

bool Shape::operator==(const Shape &other) const {
  if (this->DimensionsSize() != other.DimensionsSize()) return false;
  Index i = 0;
  for (; i < dims_.ArraySize() && dims_[i] == other.dims_[i]; ++i)
    ;
  return i == dims_.ArraySize();
}
}  // namespace KD