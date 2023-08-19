#ifndef MEMORY_POOL_KD_ALLOCATOR_H
#define MEMORY_POOL_KD_ALLOCATOR_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <utility>

#ifdef NDEBUG
#define LOG_MP_INFO(x)
#define LOG_MP_ERROR(x)
#define LOG_MP_WARNING(x)
#define LOG_MP_FATAL(x)
#else
#define LOG_MP_INFO(x)
#define LOG_MP_ERROR(x)
#define LOG_MP_WARNING(x)
#define LOG_MP_FATAL(x)
#endif

namespace KD {
namespace Allocator {
using MemorySize = unsigned long long;
struct AllocatorInterface {
  static void *Allocate(MemorySize n_bytes);
  static void Deallocate(void *ptr);
};

template <typename T>
struct DeleteHandler {
  void operator()(void *ptr) {
    LOG_MP_INFO("Release memory address:[" << ptr << "]");
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

  LOG_MP_INFO("Allocator byte size:[" << n_bytes << "] memory address:["
                                      << raw_ptr << "]");
  return SharedPtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}

// n_bytes = n_objects * sizeof(T).
template <typename T>
static UniquePtr<T> UniqueAllocate(MemorySize n_bytes) {
  void *raw_ptr = AllocatorInterface::Allocate(n_bytes);
  LOG_MP_INFO("Allocator byte size:[" << n_bytes << "] memory address:["
                                      << raw_ptr << "]");
  return UniquePtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}

template <typename T, typename... Args>
static SharedPtr<T> SharedConstruct(Args &&...args) {
  void *raw_ptr = AllocatorInterface::Allocate(sizeof(T));
  new (raw_ptr) T(std::forward<Args>(args)...);
  LOG_MP_INFO("Allocator byte size:[" << sizeof(T) << "] memory address:["
                                      << raw_ptr << "]");
  return SharedPtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}

template <typename T, typename... Args>
static UniquePtr<T> UniqueConstruct(Args &&...args) {
  void *raw_ptr = AllocatorInterface::Allocate(sizeof(T));
  new (raw_ptr) T(std::forward<Args>(args)...);
  LOG_MP_INFO("Allocator byte size:[" << sizeof(T) << "] memory address:["
                                      << raw_ptr << "]");
  return UniquePtr<T>(static_cast<T *>(raw_ptr), DeleteHandler<T>());
}
}  // namespace Allocator
}  // namespace KD

#endif