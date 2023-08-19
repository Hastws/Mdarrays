#include "learning/init.h"

#include <chrono>
#include <cmath>

namespace KD {
namespace Learning {
std::default_random_engine InitializerBase::engine_(
    std::chrono::system_clock::now().time_since_epoch().count());

CpyInitializer::CpyInitializer(Mdarray &param, BasicData *data)
    : InitializerBase(param), data_(data) {}

void CpyInitializer::Init() const {
  BasicData *storage_data_ptr = GetStorage();
  Index data_size = DataSize();
  for (Index i = 0; i < data_size; ++i) storage_data_ptr[i] = data_[i];
}

KaimingInitializer::KaimingInitializer(Mdarray &param, Mode mode,
                                       bool conv_weight)
    : InitializerBase(param), mode_(mode), conv_weight_(conv_weight) {}

void KaimingInitializer::Init() const {
  Index fan = mode_ == Mode::FAN_IN ? param_.Size(1) : param_.Size(0);
  BasicData gain = std::sqrt(2.0);
  BasicData bound = gain * std::sqrt(3.0 / fan);
  std::uniform_real_distribution<BasicData> u(-bound, bound);

  BasicData *storage_data_ptr = GetStorage();
  Index data_size = DataSize();

  for (Index i = 0; i < data_size; ++i) {
    storage_data_ptr[i] = u(engine_);
  }
}

UniformInitializer::UniformInitializer(Mdarray &param,
                                       BasicData a, BasicData b)
    : InitializerBase(param), a_(a), b_(b) {}

void UniformInitializer::Init() const {
  std::uniform_real_distribution<BasicData> u(a_, b_);
  BasicData *storage_data_ptr = GetStorage();
  Index data_size = DataSize();
  for (Index i = 0; i < data_size; ++i) {
    storage_data_ptr[i] = u(engine_);
  }
}

}  // namespace Learning
}  // namespace KD