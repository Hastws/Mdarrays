#include <cstring>

#include "learning/optimizer.h"
#include "multidimensional_arrays/multidimensional_arrays.h"
#include "multidimensional_arrays/multidimensional_arrays_impl.h"
#include "multidimensional_arrays/storage.h"

namespace KD {
namespace Learning {

OptimizerBase::OptimizerBase(const ParamsDict &params_dict) {
  params_.reserve(params_dict.size());
  for (const auto &named_param_ref : params_dict) {
    MultidimensionalArrays &ma = named_param_ref.second.get();
    auto &impl = const_cast<MultidimensionalArraysImpl &>(ma.Impl());
    CHECK_TRUE(impl.IsContiguous(),
               "Only contiguous MultidimensionalArrays can be optimized.");
    params_.emplace_back(impl);
  }
}

void OptimizerBase::ZeroGrad() {
  for (MultidimensionalArraysImpl &t : params_) {
    BasicData *grad_data_ptr = GetGrad(t);
    std::memset(grad_data_ptr, 0, t.shape_.SpaceSize() * sizeof(BasicData));
  }
}

StochasticGradientDescent::StochasticGradientDescent(
    const ParamsDict &params_dict, BasicData lr)
    : OptimizerBase(params_dict), lr_(lr) {}

void StochasticGradientDescent::Step() {
  for (MultidimensionalArraysImpl &t : params_) {
    BasicData *storage_data_ptr = GetStorage(t);
    BasicData *grad_data_ptr = GetGrad(t);
    Index data_size = DataSize(t);

    for (Index i = 0; i < data_size; ++i) {
      storage_data_ptr[i] -= lr_ * grad_data_ptr[i];
    }
  }
}

StochasticGradientDescentWithMomentum::StochasticGradientDescentWithMomentum(
    const ParamsDict &params_dict, BasicData lr, BasicData momentum)
    : OptimizerBase(params_dict),
      lr_(lr),
      momentum_(momentum),
      first_step_(true) {
  running_means_.reserve(params_.size());
  for (MultidimensionalArraysImpl &t : params_) {
    Index n_bytes = sizeof(BasicData) * DataSize(t);
    running_means_.emplace_back(Allocator::UniqueAllocate<BasicData>(n_bytes));
  }
}

void StochasticGradientDescentWithMomentum::Step() {
  if (first_step_) {
    first_step_ = false;
    for (Index i = 0; i < params_.size(); ++i) {
      MultidimensionalArraysImpl &t = params_[i];
      BasicData *storage_data_ptr = GetStorage(t);
      BasicData *grad_data_ptr = GetGrad(t);
      BasicData *vx = running_means_[i].get();
      Index data_size = DataSize(t);

      std::memcpy(vx, grad_data_ptr, data_size * sizeof(BasicData));
      for (Index j = 0; j < data_size; ++j) {
        storage_data_ptr[j] -= lr_ * vx[j];
      }
    }
  } else {
    for (Index i = 0; i < params_.size(); ++i) {
      MultidimensionalArraysImpl &t = params_[i];
      BasicData *storage_data_ptr = GetStorage(t);
      BasicData *grad_data_ptr = GetGrad(t);
      BasicData *vx = running_means_[i].get();
      Index data_size = DataSize(t);

      for (Index j = 0; j < data_size; ++j) {
        vx[j] = momentum_ * vx[j] + grad_data_ptr[j];
        storage_data_ptr[j] -= lr_ * vx[j];
      }
    }
  }
}

}  // namespace Learning
}  // namespace KD