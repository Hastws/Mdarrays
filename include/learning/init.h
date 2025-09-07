#ifndef MULTIDIMENSIONAL_ARRAYS_LEARNING_INIT_H
#define MULTIDIMENSIONAL_ARRAYS_LEARNING_INIT_H

#include <random>
#include <string>

#include "mdarray/mdarray.h"
#include "mdarray/mdarray_impl.h"
#include "utils/exception.h"

namespace Autoalg {
namespace Learning {
class InitializerBase {
 public:
  virtual ~InitializerBase() = default;
  explicit InitializerBase(Mdarray &param)
      : param_(const_cast<MdarrayImpl &>(param.Impl())) {
    CHECK_TRUE(param_.IsContiguous(),
               "Only contiguous multidimensional arrays can be initialized.");
  }

  virtual void Init() const = 0;

 protected:
  Index DataSize() const { return param_.Size().SpaceSize(); }

  BasicData *GetStorage() const {
    StorageUniversalAgent storage_universal_agent(param_.GetStorage());
    return storage_universal_agent.GetStorageData();
  }

  static std::default_random_engine engine_;
  MdarrayImpl &param_;
};

class CpyInitializer : public InitializerBase {
 public:
  CpyInitializer(Mdarray &param, BasicData *data);

  void Init() const override;

 private:
  BasicData *data_;
};

class KaimingInitializer : public InitializerBase {
 public:
  enum class Mode { FAN_IN = 0, FAN_OUT = 1 };

  explicit KaimingInitializer(Mdarray &param, Mode mode = Mode::FAN_IN,
                              bool conv_weight = false);

  void Init() const override;

 private:
  Mode mode_;
  bool conv_weight_;
};

class UniformInitializer : public InitializerBase {
 public:
  explicit UniformInitializer(Mdarray &param, BasicData a = 0.,
                              BasicData b = 1.);

  void Init() const override;

 private:
  BasicData a_;
  BasicData b_;
};

}  // namespace Learning
}  // namespace Autoalg
#endif