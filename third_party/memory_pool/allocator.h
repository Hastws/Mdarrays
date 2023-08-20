#ifndef MEMORY_POOL_ALLOCATOR_H
#define MEMORY_POOL_ALLOCATOR_H

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <utility>

namespace KD {
namespace Allocator {
using MemorySize = std::uint64_t;
struct AllocatorInterface {
  static void *Allocate(MemorySize n_bytes);
  static void Deallocate(void *ptr);
};

template <typename T>
struct DeleteHandler {
  void operator()(void *ptr) {
    static_cast<T *>(ptr)->~T();
    AllocatorInterface::Deallocate(ptr);
  }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, DeleteHandler<T>>;
template <typename T>
using SharedPtr = std::shared_ptr<T>;

// n_bytes = n_objects * sizeof(T).
template <typename T>
static SharedPtr<T> SharedAllocate(MemorySize n_bytes) {
  void *raw_ptr = AllocatorInterface::Allocate(n_bytes);
  return SharedPtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}

// n_bytes = n_objects * sizeof(T).
template <typename T>
static UniquePtr<T> UniqueAllocate(MemorySize n_bytes) {
  void *raw_ptr = AllocatorInterface::Allocate(n_bytes);
  return UniquePtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}

template <typename T, typename... Args>
static SharedPtr<T> SharedConstruct(Args &&...args) {
  void *raw_ptr = AllocatorInterface::Allocate(sizeof(T));
  new (raw_ptr) T(std::forward<Args>(args)...);
  return SharedPtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}

template <typename T, typename... Args>
static UniquePtr<T> UniqueConstruct(Args &&...args) {
  void *raw_ptr = AllocatorInterface::Allocate(sizeof(T));
  new (raw_ptr) T(std::forward<Args>(args)...);
  return UniquePtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}
}  // namespace Allocator
}  // namespace KD

#endif