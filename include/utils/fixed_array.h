#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_FIXED_ARRAY_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_FIXED_ARRAY_H

#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>

#include "memory_pool/allocator.h"
#include "utils/base_config.h"
#include "dynamic_array.h"

namespace Autoalg {

template <typename DataType>
class FixedArray {
 public:
  explicit FixedArray(Index size) : size_(size), array_{0} {}

  FixedArray(std::initializer_list<DataType> data) : FixedArray(data.size()) {
    auto p = array_;
    for (auto d : data) {
      *p = d;
      ++p;
    }
  }

  FixedArray(const FixedArray<DataType> &other)
      : FixedArray(other.ArraySize()) {
    for (Index index = 0; index < other.size_; index++) {
      array_[index] = other.array_[index];
    }
  }

  FixedArray(const DataType *data, Index size) : FixedArray(size) {
    for (Index index = 0; index < size; index++) {
      array_[index] = data[index];
    }
  }

  FixedArray(FixedArray<DataType> &&other) noexcept
      : FixedArray(other.ArraySize()) {
    for (Index index = 0; index < other.size_; index++) {
      array_[index] = other.array_[index];
    }
  }

  DataType &operator[](Index idx) { return array_[idx]; }

  DataType operator[](Index idx) const { return array_[idx]; }

  Index ArraySize() const { return size_; }

  void Memset(DataType value) {
    for (Index index = 0; index < size_; index++) {
      array_[index] = value;
    }
  }

 private:
  Index size_;
  DataType array_[FixedArraySize];
};

}  // namespace Autoalg

#endif