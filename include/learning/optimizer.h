#ifndef MULTIDIMENSIONAL_ARRAYS_LEARNING_OPTIMIZER_H
#define MULTIDIMENSIONAL_ARRAYS_LEARNING_OPTIMIZER_H

#include <functional>
#include <vector>

#include "learning/module.h"

namespace Autoalg {
namespace Learning {
class OptimizerBase {
 public:
  explicit OptimizerBase(const ParamsDict &params_dict);
  void ZeroGrad();
  virtual void Step() = 0;

 protected:
  static Index GetDataSize(const MdarrayImpl &mdarray_impl) {
    return mdarray_impl.Size().SpaceSize();
  }
  static BasicData *GetStorageData(MdarrayImpl &mdarray_impl) {
    StorageUniversalAgent storage_universal_agent(mdarray_impl.GetStorage());
    return storage_universal_agent.GetStorageData();
  };
  static BasicData *GetGradData(MdarrayImpl &mdarray_impl) {
    MdarrayImplUniversalAgent mdarray_impl_universal_agent(mdarray_impl);
    auto &storage = mdarray_impl_universal_agent.GetGradMetaPtr()->grad_;
    StorageUniversalAgent storage_universal_agent(storage);
    return storage_universal_agent.GetStorageData();
  }

  std::vector<std::reference_wrapper<MdarrayImpl>> params_;
};

class StochasticGradientDescent : public OptimizerBase {
 public:
  StochasticGradientDescent(const ParamsDict &params_dict, BasicData lr);
  void Step() override;
  BasicData Lr() const { return lr_; }
  void SetLr(BasicData lr) { lr_ = lr; }

 private:
  BasicData lr_;
};

class StochasticGradientDescentWithMomentum : public OptimizerBase {
 public:
  StochasticGradientDescentWithMomentum(const ParamsDict &params_dict,
                                        BasicData lr, BasicData momentum);
  void Step() override;
  BasicData Lr() const { return lr_; }
  void SetLr(BasicData lr) { lr_ = lr; }
  void LrDecay(BasicData factor) { lr_ *= factor; }

 private:
  BasicData lr_;
  BasicData momentum_;
  bool first_step_;
  std::vector<Allocator::UniquePtr<BasicData>> running_means_;
};
}  // namespace Learning
}  // namespace Autoalg
#endif