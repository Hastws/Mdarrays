#include <chrono>
#include <iostream>

#include "data/data.h"
#include "exp/function.h"
#include "learning/module.h"
#include "learning/optimizer.h"
#include "multidimensional_arrays/multidimensional_arrays.h"

class SimpleCNN : public KD::Learning::Module {
 public:
  KD::MultidimensionalArrays Forward(
      const KD::MultidimensionalArrays &input) override {
    KD::MultidimensionalArrays conv_0 = conv_0_.Forward(input);
    KD::MultidimensionalArrays conv_1 = conv_1_.Forward(conv_0);

    KD::MultidimensionalArrays feat(conv_1.Size());
    feat = conv_1;
    KD::MultidimensionalArrays linear_0 =
        linear_0_.Forward(feat.View({feat.Size(0), 16 * 28 * 28}));
    KD::MultidimensionalArrays linear_1 = linear_1_.Forward(linear_0);
    return linear_1;
  }

  KD::Learning::ParamsDict Parameters() override {
    return {{"conv_0_", conv_0_.Parameters()},
            {"conv_1_", conv_1_.Parameters()},
            {"linear_0_", linear_0_.Parameters()},
            {"linear_1_", linear_1_.Parameters()}};
  }

 private:
  KD::Learning::Conv2dWithReLU conv_0_{1, 6, {3, 3}, {1, 1}, {1, 1}};
  KD::Learning::MaxPool2d pool_0_{{2, 2}, {2, 2}, {0, 0}};
  KD::Learning::Conv2dWithReLU conv_1_{6, 16, {3, 3}, {1, 1}, {1, 1}};
  KD::Learning::MaxPool2d pool_1_{{2, 2}, {2, 2}, {0, 0}};
  KD::Learning::LinearWithReLU linear_0_{16 * 28 * 28, 64};
  KD::Learning::Linear linear_1_{64, 10};
};

int main() {
  // config
  constexpr KD::Index epoch = 16;
  constexpr KD::Index batch_size = 64;
  constexpr KD::BasicData lr = 0.01;
  constexpr KD::BasicData momentum = 0.9;

  constexpr KD::BasicData lr_decay_factor = 0.1;
  constexpr KD::Index lr_decay_epoch1 = 3;
  constexpr KD::Index lr_decay_epoch2 = 5;

  constexpr KD::Index print_iterators = 10;

  using namespace std::chrono;
  steady_clock::time_point start_tp = steady_clock::now();

  // dataset
  KD::SourceData::MNIST train_dataset(
      std::string(PROJECT_SOURCE_DIR) + "/data/train-images.idx3-ubyte",
      std::string(PROJECT_SOURCE_DIR) + "/data/train-labels.idx1-ubyte",
      batch_size, false);
  KD::SourceData::MNIST val_dataset(
      std::string(PROJECT_SOURCE_DIR) + "/data/t10k-images.idx3-ubyte",
      std::string(PROJECT_SOURCE_DIR) + "/data/t10k-labels.idx1-ubyte",
      batch_size, false);

  // model and criterion
  SimpleCNN simple_cnn;
  KD::Learning::CrossEntropy criterion;

  // optimizer
  KD::Learning::StochasticGradientDescentWithMomentum optimizer(
      simple_cnn.Parameters(), lr, momentum);

  KD::Index n_samples;
  const KD::BasicData *batch_samples;
  const KD::Index *batch_labels;
  for (KD::Index i = 0; i < epoch; ++i) {
    std::cout << "Epoch " << i << " training..." << std::endl;
    std::cout << "total iterators: " << train_dataset.BatchesSize()
              << std::endl;
    train_dataset.Shuffle();

    if (i == lr_decay_epoch1 || i == lr_decay_epoch2) {
      KD::BasicData optimizer_lr = optimizer.Lr();
      optimizer.SetLr(optimizer_lr * lr_decay_factor);
      std::cout << "Lr decay to " << optimizer.Lr() << std::endl;
    }

    for (KD::Index j = 0; j < train_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          train_dataset.GetBatch(j);
      KD::MultidimensionalArrays input(batch_samples, {n_samples, 1, 28, 28});

      KD::MultidimensionalArrays output = simple_cnn.Forward(input);
      KD::MultidimensionalArrays loss = criterion.Forward(output, batch_labels);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      if (j % print_iterators == 0) {
        std::cout << "iter " << j << " | ";
        std::cout << "loss: " << loss.Item() << std::endl;
      }
    }

    std::cout << "Epoch " << i << " Evaluating..." << std::endl;
    KD::Index total_samples = 0, correct_samples = 0;
    for (KD::Index j = 0; j < val_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          val_dataset.GetBatch(j);
      KD::MultidimensionalArrays input(batch_samples, {n_samples, 1, 28, 28});

      KD::MultidimensionalArrays output = simple_cnn.Forward(input);
      KD::MultidimensionalArrays predict =
          KD::Operator::CreateOperationArgmax(output, 1);
      for (KD::Index k = 0; k < n_samples; ++k) {
        ++total_samples;
        KD::Index pd_label = static_cast<KD::Index>(predict[{k}]);
        if (pd_label == batch_labels[k]) ++correct_samples;
      }
    }
    std::cout << "total samples: " << total_samples;
    std::cout << " | correct samples: " << correct_samples;
    std::cout << " | acc: ";
    std::cout << static_cast<KD::BasicData>(correct_samples) / total_samples
              << std::endl;
  }

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  std::cout << "Training finished. Training took " << time_span.count();
  std::cout << " seconds." << std::endl;
  return 0;
}
