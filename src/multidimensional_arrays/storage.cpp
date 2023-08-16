#include "multidimensional_arrays/storage.h"

#include <cstring>

namespace KD {

Storage::Storage(Index Size)
    : base_ptr_(Allocator::SharedAllocate<VersionData>(
          Size * sizeof(BasicData) + sizeof(Index))),
      data_ptr_(base_ptr_->data_) {
  base_ptr_->version_ = 0;
}

Storage::Storage(const Storage &other, Index offset)
    : base_ptr_(other.base_ptr_), data_ptr_(other.data_ptr_ + offset) {}

Storage::Storage(Index Size, BasicData value) : Storage(Size) {
  std::memset(data_ptr_, static_cast<int>(value), Size * sizeof(BasicData));
}

Storage::Storage(const BasicData *data, Index Size) : Storage(Size) {
  std::memcpy(data_ptr_, data, Size * sizeof(BasicData));
}

}  // namespace KD