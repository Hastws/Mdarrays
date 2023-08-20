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
    static constexpr Index rows_size_ = 28;
    static constexpr Index cols_size_ = 28;
    static constexpr Index pixels_size_ = rows_size_ * cols_size_;
    BasicData pixels_data_[pixels_size_];
  };

  MNIST(const std::string &img_path, const std::string &label_path,
        Index batch_size);

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
    static constexpr Index channels_size_ = 3;
    static constexpr Index rows_size_ = 32;
    static constexpr Index cols_size_ = 32;
    static constexpr Index pixels_size_ =
        channels_size_ * rows_size_ * cols_size_;

    static constexpr Index train_samples_size_ = 50000;
    static constexpr Index test_samples_size_ = 10000;

    BasicData pixels_data_[pixels_size_];
  };

  Cifar10(const std::string &dataset_dir, bool train, Index batch_size);
  Index SamplesSize() const override { return samples_size_.size(); }
  Index BatchesSize() const override { return batches_size_; }

  std::pair<const BasicData *, Index> GetSample(Index idx) const override;
  std::tuple<Index, const BasicData *, const Index *> GetBatch(
      Index idx) const override;
  void Shuffle() override;

 private:
  void ReadCifar10(const std::string &dataset_dir, bool train);
  void ReadBin(const std::string &bin_path);

  Index batch_size_;
  Index batches_size_;
  std::vector<Img> samples_size_;
  std::vector<Index> labels_;
};

}  // namespace SourceData
}  // namespace KD
#endif