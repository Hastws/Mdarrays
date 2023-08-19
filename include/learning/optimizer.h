#ifndef MULTIDIMENSIONAL_ARRAYS_LEARNING_OPTIMIZER_H
#define MULTIDIMENSIONAL_ARRAYS_LEARNING_OPTIMIZER_H

#include <functional>
#include <vector>

#include "learning/module.h"

namespace KD {
namespace Learning {
class OptimizerBase {
 public:
  explicit OptimizerBase(const ParamsDict &params_dict);
  void ZeroGrad();
  virtual void Step() = 0;

 protected:
  static Index DataSize(const MdarrayImpl &t) { return t.shape_.SpaceSize(); }
  static BasicData *GetStorage(MdarrayImpl &t) { return t.storage_.data_ptr_; };
  static BasicData *GetGrad(MdarrayImpl &t) {
    return t.grad_meta_ptr_->grad_.data_ptr_;
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
}  // namespace KD
#endif