#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MULTIDIMENSIONAL_ARRAYS_SHAPE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MULTIDIMENSIONAL_ARRAYS_SHAPE_H

#include <initializer_list>
#include <ostream>

#include "memory_pool/allocator.h"
#include "utils/array.h"
#include "utils/base_config.h"

namespace KD {

class Shape {
 public:
  // constructor
  Shape(std::initializer_list<Index> dims);

  Shape(const Shape &other, Index skip);

  Shape(Index *dims, Index dim);

  Shape(IndexArray &&shape);

  Shape(const Shape &other) = default;

  Shape(Shape &&other) = default;

  ~Shape() = default;

  // method
  Index SpaceSize() const;

  Index SubSpaceSize(Index start_dim, Index end_dim) const;

  Index SubSpaceSize(Index start_dim) const;

  bool operator==(const Shape &other) const;

  // inline function
  Index DimensionsSize() const { return dims_.ArraySize(); }

  Index operator[](Index idx) const { return dims_[idx]; }

  Index &operator[](Index idx) { return dims_[idx]; }

  explicit operator const IndexArray &() const { return dims_; }

  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const Shape &s);

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