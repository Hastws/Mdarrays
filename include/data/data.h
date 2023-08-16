#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_DATA_DATA_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_DATA_DATA_H

#include <string>
#include <tuple>
#include <vector>

#include "utils/base_config.h"

namespace KD {
namespace SourceData {
class DatasetBase {
 public:
  virtual Index SamplesSize() const = 0;
  virtual Index BatchesSize() const = 0;

  virtual std::pair<const BasicData *, Index> GetSample(Index idx) const = 0;
  virtual std::tuple<Index, const BasicData *, const Index *> GetBatch(
      Index idx) const = 0;
  virtual void Shuffle() = 0;
};

class MNIST : public DatasetBase {
 public:
  struct Img {
    static constexpr Index n_rows_ = 28;
    static constexpr Index n_cols_ = 28;
    static constexpr Index n_pixels_ = n_rows_ * n_cols_;
    BasicData pixels_[n_pixels_];
  };

  MNIST(const std::string &img_path, const std::string &label_path,
        Index batch_size, bool shuffle);

  Index SamplesSize() const override { return samples_size_.size(); }
  Index BatchesSize() const override { return batches_size_; }

  std::pair<const BasicData *, Index> GetSample(Index idx) const override;
  std::tuple<Index, const BasicData *, const Index *> GetBatch(
      Index idx) const override;
  void Shuffle() override;

 private:
  void ReadMnistImages(const std::string &path);
  void ReadMnistLabels(const std::string &path);

  Index batch_size_;
  Index batches_size_;
  std::vector<Img> samples_size_;
  std::vector<Index> labels_;
};

class Cifar10 : public DatasetBase {
 public:
  struct Img {
    static constexpr Index n_channels_ = 3;
    static constexpr Index n_rows_ = 32;
    static constexpr Index n_cols_ = 32;
    static constexpr Index n_pixels_ = n_channels_ * n_rows_ * n_cols_;

    static constexpr Index n_train_samples_ = 50000;
    static constexpr Index n_test_samples_ = 10000;

    BasicData data_[n_pixels_];
  };

  Cifar10(const std::string &dataset_dir, bool train, Index batch_size,
          bool shuffle, char path_sep = '\\');
  Index SamplesSize() const override { return imgs_.size(); }
  Index BatchesSize() const override { return n_batchs_; }

  std::pair<const BasicData *, Index> GetSample(Index idx) const override;
  std::tuple<Index, const BasicData *, const Index *> GetBatch(
      Index idx) const override;
  void Shuffle() override;

 private:
  void ReadCifar10(const std::string &dataset_dir, bool train,
                   char path_sep = '\\');
  void ReadBin(const std::string &bin_path);

  Index batch_size_;
  Index n_batchs_;
  std::vector<Img> imgs_;
  std::vector<Index> labels_;
};

}  // namespace SourceData
}  // namespace KD
#endif