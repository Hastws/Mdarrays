#include <chrono>

#include "data/data.h"
#include "exp/function.h"
#include "learning/module.h"
#include "learning/optimizer.h"
#include "mdarray/mdarray.h"

class MLP : public Autoalg::Learning::Module {
 public:
  MLP(Autoalg::Index in, Autoalg::Index hidden_1, Autoalg::Index out)
      : linear_1_(in, hidden_1), linear_2_(hidden_1, out) {}

  Autoalg::Mdarray Forward(const Autoalg::Mdarray &input) override {
    Autoalg::Mdarray x1 = linear_1_.Forward(input);
    Autoalg::Mdarray y = linear_2_.Forward(x1);
    return y;
  }

  Autoalg::Learning::ParamsDict Parameters() override {
    return {{"linear_1", linear_1_.Parameters()},
            {"linear_2", linear_2_.Parameters()}};
  }

 private:
  Autoalg::Learning::LinearWithReLU linear_1_;
  Autoalg::Learning::Linear linear_2_;
};

int main() {
  constexpr Autoalg::Index epoch = 3;
  constexpr Autoalg::Index batch_size = 64;
  constexpr Autoalg::BasicData lr = 0.05;
  constexpr Autoalg::BasicData momentum = 0.90;
  constexpr Autoalg::BasicData lr_decay_factor = 0.1;
  constexpr Autoalg::Index lr_decay_epoch = 2;
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
  MLP mlp(Autoalg::SourceData::MNIST::Img::pixels_size_, 64, 10);
  Autoalg::Learning::CrossEntropy criterion;

  // optimizer
  Autoalg::Learning::StochasticGradientDescentWithMomentum optimizer(
      mlp.Parameters(), lr, momentum);

  Autoalg::Index n_samples;
  const Autoalg::BasicData *batch_samples;
  const Autoalg::Index *batch_labels;
  for (Autoalg::Index i = 0; i < epoch; ++i) {
    LOG_MDA_INFO("Epoch " << i << " training...")
    LOG_MDA_INFO("total iterations: " << train_dataset.BatchesSize())
    train_dataset.Shuffle();

    if (i == lr_decay_epoch) {
      Autoalg::BasicData optimizer_lr = optimizer.Lr();
      optimizer.SetLr(optimizer_lr * lr_decay_factor);
      LOG_MDA_INFO("Lr decay to " << optimizer.Lr())
    }

    for (Autoalg::Index j = 0; j < train_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          train_dataset.GetBatch(j);
      Autoalg::Mdarray input(batch_samples,
                        {n_samples, Autoalg::SourceData::MNIST::Img::pixels_size_});

      Autoalg::Mdarray output = mlp.Forward(input);
      Autoalg::Mdarray loss = criterion.Forward(output, batch_labels);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      if (j % print_iterators == 0) {
        LOG_MDA_INFO("iter " << j << " | loss: " << loss.Item())
      }
    }

    LOG_MDA_INFO("Epoch " << i << " Evaluating...")
    Autoalg::Index total_samples = 0, correct_samples = 0;
    for (Autoalg::Index j = 0; j < val_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          val_dataset.GetBatch(j);
      Autoalg::Mdarray input(batch_samples,
                        {n_samples, Autoalg::SourceData::MNIST::Img::pixels_size_});
      Autoalg::Mdarray output = mlp.Forward(input);
      Autoalg::Mdarray predict = Autoalg::Operator::CreateOperationArgmax(output, 1);

      for (Autoalg::Index k = 0; k < n_samples; ++k) {
        ++total_samples;
        Autoalg::Index pd_label = static_cast<Autoalg::Index>(predict[{k}]);
        if (pd_label == batch_labels[k]) {
          ++correct_samples;
        }
      }
    }
    LOG_MDA_INFO("total samples: " << total_samples << " | correct samples: "
                                   << correct_samples << " | acc: ")
    LOG_MDA_INFO(static_cast<Autoalg::BasicData>(correct_samples) / total_samples)
  }

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  LOG_MDA_INFO("Training finished. Training took " << time_span.count()
                                                   << " seconds.")
  return 0;
}
