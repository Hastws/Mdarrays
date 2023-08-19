#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_EXCEPTION_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_EXCEPTION_H

#include <algorithm>
#include <cstdio>
#include <exception>

#include "log.h"

namespace KD {
struct Error : public std::exception {
  Error(const char *file, const char *func, unsigned int line);

  const char *what() const noexcept override;

  static char msg_[1024];
  const char *file_;
  const char *func_;
  const unsigned int line_;
};

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(msg)               \
  do {                                 \
    LOG_MDA_ERROR(msg)                 \
    throw ::KD::Error(ERROR_LOCATION); \
  } while (0)

#ifndef NDEBUG
// base assert macro
#define CHECK_TRUE(expr, msg) \
  if (!(expr)) THROW_ERROR(msg)

#define CHECK_NOT_NULL(ptr, msg) \
  if (nullptr == (ptr)) THROW_ERROR(msg)

#define CHECK_EQUAL(x, y, msg) \
  if ((x) != (y)) THROW_ERROR(msg)

#define CHECK_IN_RANGE(x, lower, upper, msg) \
  if ((x) < (lower) || (x) >= (upper)) THROW_ERROR(msg)

#define CHECK_FLOAT_EQUAL(x, y, msg) \
  if (std::abs((x) - (y)) > 1e-4) THROW_ERROR(msg)

#define CHECK_INDEX_VALID(x, msg) \
  if ((x) > INDEX_MAX) THROW_ERROR(msg)

// assert macro only working for ExpImpl
#define CHECK_EXP_SAME_SHAPE(e1_, e2_)                                        \
  do {                                                                        \
    auto &e1 = (e1_);                                                         \
    auto &e2 = (e2_);                                                         \
    CHECK_EQUAL(e1.DimensionsSize(), e2.DimensionsSize(),                     \
                "Expect the same dimensions, but got "                        \
                    << e1.DimensionsSize() << " and " << e2.DimensionsSize()  \
                    << ".");                                                  \
    for (Index i = 0; i < e1.DimensionsSize(); ++i)                           \
      CHECK_EQUAL(e1.Size(i), e2.Size(i),                                     \
                  "Expect the same Size on the "                              \
                      << i << " dimension, but got " << e1.Size(i) << " and " \
                      << e2.Size(i) << ".");                                  \
  } while (0)

#define CHECK_EXP_BROADCAST(e1_, e2_)                                      \
  do {                                                                     \
    auto &e1 = (e1_);                                                      \
    auto &e2 = (e2_);                                                      \
    Index min_dim = std::min(e1.DimensionsSize(), e2.DimensionsSize());    \
    for (Index i = 0; i < min_dim; ++i)                                    \
      CHECK_TRUE(                                                          \
          e1.Size(i) == e2.Size(i) || e1.Size(i) == 1 || e2.Size(i) == 1,  \
          "The Size on " << i << "th dimension, " << e1.Size(i) << " and " \
                         << e2.Size(i) << ", can't be broad casted.");     \
  } while (0)

#else

#define CHECK_TRUE(expr, msg) \
  {}
#define CHECK_NOT_NULL(ptr, msg) \
  {}
#define CHECK_EQUAL(x, y, msg) \
  {}
#define CHECK_IN_RANGE(x, lower, upper, msg) \
  {}
#define CHECK_FLOAT_EQUAL(x, y, msg) \
  {}
#define CHECK_INDEX_VALID(x, msg) \
  {}
#define CHECK_EXP_SAME_SHAPE(e1, e2) \
  {}
#define CHECK_EXP_BROADCAST(e1, e2) \
  {}

#endif

}  // namespace KD
#endif
