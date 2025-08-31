#include <chrono>
#include <iostream>

#include "data/data.h"
#include "exp/function.h"
#include "learning/module.h"
#include "learning/optimizer.h"
#include "mdarray/mdarray.h"

class CNN : public Autoalg::Learning::Module {
 public:
  Autoalg::Mdarray Forward(const Autoalg::Mdarray &input) override {
    Autoalg::Mdarray conv_0 = conv_0_.Forward(input);
    Autoalg::Mdarray conv_1 = conv_1_.Forward(conv_0);

    Autoalg::Mdarray feat(conv_1.Size());
    feat = conv_1;
    Autoalg::Mdarray linear_0 =
        linear_0_.Forward(feat.View({feat.Size(0), 16 * 28 * 28}));
    Autoalg::Mdarray linear_1 = linear_1_.Forward(linear_0);
    return linear_1;
  }

  Autoalg::Learning::ParamsDict Parameters() override {
    return {{"conv_0_", conv_0_.Parameters()},
            {"conv_1_", conv_1_.Parameters()},
            {"linear_0_", linear_0_.Parameters()},
            {"linear_1_", linear_1_.Parameters()}};
  }

 private:
  Autoalg::Learning::Conv2dWithReLU conv_0_{1, 6, {3, 3}, {1, 1}, {1, 1}};
  Autoalg::Learning::MaxPool2d pool_0_{{2, 2}, {2, 2}, {0, 0}};
  Autoalg::Learning::Conv2dWithReLU conv_1_{6, 16, {3, 3}, {1, 1}, {1, 1}};
  Autoalg::Learning::MaxPool2d pool_1_{{2, 2}, {2, 2}, {0, 0}};
  Autoalg::Learning::LinearWithReLU linear_0_{16 * 28 * 28, 64};
  Autoalg::Learning::Linear linear_1_{64, 10};
};

int main() {
  // config
  constexpr Autoalg::Index epoch = 16;
  constexpr Autoalg::Index batch_size = 64;
  constexpr Autoalg::BasicData lr = 0.01;
  constexpr Autoalg::BasicData momentum = 0.9;

  constexpr Autoalg::BasicData lr_decay_factor = 0.1;
  constexpr Autoalg::Index lr_decay_epoch1 = 3;
  constexpr Autoalg::Index lr_decay_epoch2 = 5;

  constexpr Autoalg::Index print_iterators = 10;

  using namespace std::chrono;
  steady_clock::time_point start_tp = steady_clock::now();

  // dataset
  Autoalg::SourceData::MNIST train_dataset(std::string(MLP_MNIST_TRAIN_IMAGES),
                                      std::string(MLP_MNIST_TRAIN_LABELS),
                                      batch_size);
  Autoalg::SourceData::MNIST val_dataset(std::string(MLP_MNIST_TEST_IMAGES),
                                    std::string(MLP_MNIST_TEST_LABELS),
                                    batch_size);

  // model and criterion
  CNN simple_cnn;
  Autoalg::Learning::CrossEntropy criterion;

  // optimizer
  Autoalg::Learning::StochasticGradientDescentWithMomentum optimizer(
      simple_cnn.Parameters(), lr, momentum);

  Autoalg::Index n_samples;
  const Autoalg::BasicData *batch_samples;
  const Autoalg::Index *batch_labels;
  for (Autoalg::Index i = 0; i < epoch; ++i) {
    LOG_MDA_INFO("Epoch [" << i << "] training...")
    LOG_MDA_INFO("Total iterators: " << train_dataset.BatchesSize())
    train_dataset.Shuffle();

    if (i == lr_decay_epoch1 || i == lr_decay_epoch2) {
      Autoalg::BasicData optimizer_lr = optimizer.Lr();
      optimizer.SetLr(optimizer_lr * lr_decay_factor);
      LOG_MDA_INFO("Lr decay to [" << optimizer.Lr() << "]")
    }

    for (Autoalg::Index j = 0; j < train_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          train_dataset.GetBatch(j);
      Autoalg::Mdarray input(batch_samples, {n_samples, 1, 28, 28});

      Autoalg::Mdarray output = simple_cnn.Forward(input);
      Autoalg::Mdarray loss = criterion.Forward(output, batch_labels);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      if (j % print_iterators == 0) {
        LOG_MDA_INFO("iter [" << j << "] loss: [" << loss.Item() << "]")
      }
    }

    LOG_MDA_INFO("Epoch [" << i << "] evaluating...")
    Autoalg::Index total_samples = 0, correct_samples = 0;
    for (Autoalg::Index j = 0; j < val_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          val_dataset.GetBatch(j);
      Autoalg::Mdarray input(batch_samples, {n_samples, 1, 28, 28});

      Autoalg::Mdarray output = simple_cnn.Forward(input);
      Autoalg::Mdarray predict = Autoalg::Operator::CreateOperationArgmax(output, 1);
      for (Autoalg::Index k = 0; k < n_samples; ++k) {
        ++total_samples;
        Autoalg::Index pd_label = static_cast<Autoalg::Index>(predict[{k}]);
        if (pd_label == batch_labels[k]) ++correct_samples;
      }
    }
    LOG_MDA_INFO("total samples: ["
                 << total_samples << "] correct samples: [" << correct_samples
                 << "] acc: ["
                 << static_cast<Autoalg::BasicData>(correct_samples) / total_samples)
  }

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  LOG_MDA_INFO("Training finished training took " << time_span.count()
                                                   << " seconds.")
  return 0;
}
