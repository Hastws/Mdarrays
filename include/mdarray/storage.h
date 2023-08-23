#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_STORAGE_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_MDARRAY_STORAGE_H

#include <cstring>
#include <memory>

#include "memory_pool/allocator.h"
#include "utils/base_config.h"
#include "utils/log.h"

namespace KD {

class StorageUniversalAgent;

class Storage {
 public:
  explicit Storage(Index size)
      : base_ptr_(Allocator::SharedAllocate<VersionData>(
            size * sizeof(BasicData) + sizeof(Index))),
        data_ptr_(base_ptr_->data_) {
    base_ptr_->version_ = 0;
  }

  Storage(const Storage &other, Index offset)
      : base_ptr_(other.base_ptr_), data_ptr_(other.data_ptr_ + offset) {}

  Storage(Index size, BasicData value) : Storage(size) {
    std::memset(data_ptr_, static_cast<int>(value), size * sizeof(BasicData));
  }

  Storage(const BasicData *data, Index size) : Storage(size) {
    std::memcpy(data_ptr_, data, size * sizeof(BasicData));
  }

  Storage(const Storage &other) = default;
  Storage(Storage &&other) = default;
  ~Storage() = default;
  Storage &operator=(const Storage &other) = delete;

  // inline function
  BasicData operator[](Index idx) const { return data_ptr_[idx]; }
  BasicData &operator[](Index idx) { return data_ptr_[idx]; }

  Index Offset() const { return data_ptr_ - base_ptr_->data_; }
  Index Version() const { return base_ptr_->version_; }
  void IncrementVersion() const { ++base_ptr_->version_; }

  // friend function
  friend class StorageUniversalAgent;

 private:
  struct VersionData {
    Index version_;
    BasicData data_[1];
  };

  std::shared_ptr<VersionData> base_ptr_;  // base pointer
  BasicData *data_ptr_;                    // SourceData pointer
};

class StorageUniversalAgent {
 public:
  explicit StorageUniversalAgent(const Storage &storage) : storage_(storage) {}
  BasicData *GetStorageData() const { return storage_.data_ptr_; }

 private:
  const Storage &storage_;
};

}  // namespace KD
#endif