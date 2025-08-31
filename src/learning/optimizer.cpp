#include <cstring>

#include "learning/optimizer.h"
#include "mdarray/mdarray.h"
#include "mdarray/mdarray_impl.h"
#include "mdarray/storage.h"

namespace Autoalg {
namespace Learning {

OptimizerBase::OptimizerBase(const ParamsDict &params_dict) {
  params_.reserve(params_dict.size());
  for (const auto &named_param_ref : params_dict) {
    Mdarray &ma = named_param_ref.second.get();
    auto &impl = const_cast<MdarrayImpl &>(ma.Impl());
    CHECK_TRUE(impl.IsContiguous(),
               "Only contiguous Mdarray can be optimized.")
    params_.emplace_back(impl);
  }
}

void OptimizerBase::ZeroGrad() {
  for (MdarrayImpl &t : params_) {
    BasicData *grad_data_ptr = GetGradData(t);
    std::memset(grad_data_ptr, 0, t.Size().SpaceSize() * sizeof(BasicData));
  }
}

StochasticGradientDescent::StochasticGradientDescent(
    const ParamsDict &params_dict, BasicData lr)
    : OptimizerBase(params_dict), lr_(lr) {}

void StochasticGradientDescent::Step() {
  for (MdarrayImpl &t : params_) {
    BasicData *storage_data_ptr = GetStorageData(t);
    BasicData *grad_data_ptr = GetGradData(t);
    Index data_size = GetDataSize(t);

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
  for (MdarrayImpl &t : params_) {
    Index n_bytes = sizeof(BasicData) * GetDataSize(t);
    running_means_.emplace_back(Allocator::UniqueAllocate<BasicData>(n_bytes));
  }
}

void StochasticGradientDescentWithMomentum::Step() {
  if (first_step_) {
    first_step_ = false;
    for (Index i = 0; i < params_.size(); ++i) {
      MdarrayImpl &t = params_[i];
      BasicData *storage_data_ptr = GetStorageData(t);
      BasicData *grad_data_ptr = GetGradData(t);
      BasicData *vx = running_means_[i].get();
      Index data_size = GetDataSize(t);

      std::memcpy(vx, grad_data_ptr, data_size * sizeof(BasicData));
      for (Index j = 0; j < data_size; ++j) {
        storage_data_ptr[j] -= lr_ * vx[j];
      }
    }
  } else {
    for (Index i = 0; i < params_.size(); ++i) {
      MdarrayImpl &t = params_[i];
      BasicData *storage_data_ptr = GetStorageData(t);
      BasicData *grad_data_ptr = GetGradData(t);
      BasicData *vx = running_means_[i].get();
      Index data_size = GetDataSize(t);

      for (Index j = 0; j < data_size; ++j) {
        vx[j] = momentum_ * vx[j] + grad_data_ptr[j];
        storage_data_ptr[j] -= lr_ * vx[j];
      }
    }
  }
}

}  // namespace Learning
}  // namespace Autoalg