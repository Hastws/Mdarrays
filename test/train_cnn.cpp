#include <chrono>
#include <iostream>

#include "data/data.h"
#include "exp/function.h"
#include "learning/module.h"
#include "learning/optimizer.h"
#include "mdarray/mdarray.h"

class SimpleCNN : public KD::Learning::Module {
 public:
  SimpleCNN() = default;

  ~SimpleCNN() override = default;

  KD::Mdarray Forward(const KD::Mdarray &input) override {
    KD::Mdarray conv_0 = conv_0_.Forward(input);
    KD::Mdarray pool_0 = pool_0_.Forward(conv_0);
    KD::Mdarray conv_1 = conv_1_.Forward(pool_0);
    KD::Mdarray pool_1 = pool_1_.Forward(conv_1);

    KD::Mdarray feat(pool_1.Size());
    feat = pool_1;
    KD::Mdarray linear_0 =
        linear_0_.Forward(feat.View({feat.Size(0), 16 * 5 * 5}));
    KD::Mdarray linear_1 = linear_1_.Forward(linear_0);
    return linear_1;
  }

  KD::Learning::ParamsDict Parameters() override {
    return {{"conv_0_", conv_0_.Parameters()},
            {"pool_0_", pool_0_.Parameters()},
            {"conv_1_", conv_1_.Parameters()},
            {"pool_1_", pool_1_.Parameters()},
            {"linear_0_", linear_0_.Parameters()},
            {"linear_1_", linear_1_.Parameters()}};
  }

 private:
  KD::Learning::Conv2dWithReLU conv_0_{3, 18, {5, 5}, {1, 1}, {1, 1}};
  KD::Learning::MaxPool2d pool_0_{{2, 2}, {2, 2}, {0, 0}};
  KD::Learning::Conv2dWithReLU conv_1_{18, 18, {5, 5}, {1, 1}, {1, 1}};
  KD::Learning::MaxPool2d pool_1_{{2, 2}, {2, 2}, {0, 0}};
  KD::Learning::LinearWithReLU linear_0_{16 * 5 * 5, 84};
  KD::Learning::Linear linear_1_{84, 10};
};

int main() {
  // config
  constexpr KD::Index epoch = 7;
  constexpr KD::Index batch_size = 64;
  constexpr KD::BasicData lr = 0.01;
  constexpr KD::BasicData momentum = 0.9;

  constexpr KD::Index lr_decay_factor = 0.1;
  constexpr KD::Index lr_decay_epoch1 = 3;
  constexpr KD::Index lr_decay_epoch2 = 5;

  constexpr KD::Index print_iters = 10;

  using namespace std::chrono;
  steady_clock::time_point start_tp = steady_clock::now();

  // dataset
  KD::SourceData::Cifar10 train_dataset(
      /*dataset_dir=*/std::string(PROJECT_SOURCE_DIR) +
          "/data/cifar-10-batches-bin",
      /*train=*/true,
      /*batch_size=*/batch_size,
      /*shuffle=*/false,
      /*path_sep=*/'/');
  KD::SourceData::Cifar10 val_dataset(
      /*dataset_dir=*/std::string(PROJECT_SOURCE_DIR) +
          "/data/cifar-10-batches-bin",
      /*train=*/false,
      /*batch_size=*/batch_size,
      /*shuffle=*/false,
      /*path_sep=*/'/');
  std::cout << "train dataset length: " << train_dataset.n_samples()
            << std::endl;
  std::cout << "val dataset length: " << val_dataset.n_samples() << std::endl;

  // model and criterion
  SimpleCNN scnn;
  KD::Learning::CrossEntropy criterion;

  // optimizer
  KD::Learning::StochasticGradientDescentWithMomentum optimizer(
      scnn.Parameters(), /*Lr=*/lr, /*momentum=*/momentum);

  KD::Index n_samples;
  const KD::BasicData *batch_samples;
  const KD::Index *batch_labels;
  for (KD::Index i = 0; i < epoch; ++i) {
    std::cout << "Epoch " << i << " training..." << std::endl;
    std::cout << "total iters: " << train_dataset.n_batchs() << std::endl;
    train_dataset.shuffle();

    if (i == lr_decay_epoch1 || i == lr_decay_epoch2) {
      KD::BasicData lr = optimizer.Lr();
      optimizer.SetLr(lr * lr_decay_factor);
      std::cout << "Lr decay to " << optimizer.Lr() << std::endl;
    }

    for (KD::Index j = 0; j < train_dataset.n_batchs(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          train_dataset.get_batch(j);
      KD::Mdarray input(batch_samples,
                        {n_samples, KD::SourceData::Cifar10::Img::n_channels_,
                         KD::SourceData::Cifar10::Img::n_rows_,
                         KD::SourceData::Cifar10::Img::n_cols_});

      KD::Mdarray output = scnn.Forward(input);
      KD::Mdarray loss = criterion.Forward(output, batch_labels);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      if (j % print_iters == 0) {
        std::cout << "iter " << j << " | ";
        std::cout << "loss: " << loss.Item() << std::endl;
      }
    }

    std::cout << "Epoch " << i << " Evaluating..." << std::endl;
    KD::Index total_samples = 0, correct_samples = 0;
    for (KD::Index j = 0; j < val_dataset.n_batchs(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          val_dataset.get_batch(j);
      KD::Mdarray input(batch_samples,
                        {n_samples, KD::SourceData::Cifar10::Img::n_channels_,
                         KD::SourceData::Cifar10::Img::n_rows_,
                         KD::SourceData::Cifar10::Img::n_cols_});

      KD::Mdarray output = scnn.Forward(input);
      KD::Mdarray predict = KD::Operator::CreateOperationArgmax(output, 1);
      for (KD::Index k = 0; k < n_samples; ++k) {
        ++total_samples;
        KD::Index pd_label = predict[{k}];
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
