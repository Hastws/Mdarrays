#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_DYNAMIC_ARRAY_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_DYNAMIC_ARRAY_H

#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>

#include "memory_pool/allocator.h"
#include "utils/base_config.h"

namespace KD {

template <typename DataType>
class DynamicArray {
 public:
  explicit DynamicArray(Index Size)
      : size_(Size),
        data_ptr_(
            Allocator::UniqueAllocate<DataType>(size_ * sizeof(DataType))) {}

  DynamicArray(std::initializer_list<DataType> data)
      : DynamicArray(data.size()) {
    auto p = data_ptr_.get();
    for (auto d : data) {
      *p = d;
      ++p;
    }
  }

  DynamicArray(const DynamicArray<DataType> &other)
      : DynamicArray(other.ArraySize()) {
    std::memcpy(data_ptr_.get(), other.data_ptr_.get(),
                size_ * sizeof(DataType));
  }

  DynamicArray(const DataType *data, Index Size) : DynamicArray(Size) {
    std::memcpy(data_ptr_.get(), data, size_ * sizeof(DataType));
  }

  explicit DynamicArray(DynamicArray<DataType> &&other) = default;

  ~DynamicArray() = default;

  DataType &operator[](Index idx) { return data_ptr_.get()[idx]; }

  DataType operator[](Index idx) const { return data_ptr_.get()[idx]; }

  Index ArraySize() const { return size_; }

  void Memset(int value) const {
    std::memset(data_ptr_.get(), value, size_ * sizeof(DataType));
  }

 private:
  Index size_;
  Allocator::UniquePtr<DataType> data_ptr_;
};

}  // namespace KD

#endif