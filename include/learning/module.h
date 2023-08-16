#ifndef MULTIDIMENSIONAL_ARRAYS_LEARNING_MODULE_H
#define MULTIDIMENSIONAL_ARRAYS_LEARNING_MODULE_H

#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include "multidimensional_arrays/multidimensional_arrays.h"

namespace KD {
namespace Learning {
class ParamsDict
    : public std::unordered_map<
          std::string, std::reference_wrapper<MultidimensionalArrays>> {
 public:
  ParamsDict() = default;

  ParamsDict(std::initializer_list<value_type> items)
      : std::unordered_map<std::string,
                           std::reference_wrapper<MultidimensionalArrays>>(
            items) {}

  ParamsDict(std::initializer_list<std::pair<std::string, ParamsDict>> dicts) {
    for (auto &named_dict : dicts) {
      auto &name = named_dict.first;
      auto &dict = named_dict.second;

      for (auto &iterator : dict) {
        this->insert({name + iterator.first, iterator.second});
      }
    }
  }

  MultidimensionalArrays &operator[](const std::string &key) {
    auto iter = find(key);
    return iter->second.get();
  }
};

class Module {
 public:
  virtual MultidimensionalArrays Forward(
      const MultidimensionalArrays &input) = 0;

  virtual ParamsDict Parameters() = 0;

  virtual ~Module() = default;
};

class Linear : public Module {
 public:
  Linear(Index in_features, Index out_features);

  Linear(const Linear &other) = delete;

  ~Linear() override = default;

  MultidimensionalArrays Forward(const MultidimensionalArrays &input) override;

  ParamsDict Parameters() override;

 protected:
  MultidimensionalArrays weight_;
  MultidimensionalArrays bias_;
};

class LinearWithReLU : public Linear {
 public:
  LinearWithReLU(Index in_features, Index out_features);

  MultidimensionalArrays Forward(const MultidimensionalArrays &input) override;
};

class Conv2d : public Module {
 public:
  using MatrixSize = Operator::Img2col::MatrixSize;

  Conv2d(Index in_channels, Index out_channels, MatrixSize kernel_size,
         MatrixSize stride, MatrixSize padding);

  Conv2d(const Conv2d &other) = delete;

  ~Conv2d() override = default;

  MultidimensionalArrays Forward(const MultidimensionalArrays &input) override;

  ParamsDict Parameters() override;

 protected:
  Index in_channels_;
  Index out_channels_;

  MatrixSize kernel_size_;
  MatrixSize stride_;
  MatrixSize padding_;

  MultidimensionalArrays weight_;
};

class Conv2dWithReLU : public Conv2d {
 public:
  Conv2dWithReLU(Index in_channels, Index out_channels,
                 const MatrixSize &kernel_size, const MatrixSize &stride,
                 const MatrixSize &padding);

  MultidimensionalArrays Forward(const MultidimensionalArrays &input) override;
};

class MaxPool2d : public Module {
 public:
  using MatrixSize = Operator::Img2col::MatrixSize;

  MaxPool2d(MatrixSize kernel_size, MatrixSize stride,
            MatrixSize padding);

  MaxPool2d(const MaxPool2d &other) = delete;

  ~MaxPool2d() override = default;

  MultidimensionalArrays Forward(const MultidimensionalArrays &input) override;

  ParamsDict Parameters() override;

 protected:
  MatrixSize kernel_size_;
  MatrixSize stride_;
  MatrixSize padding_;
};

class CrossEntropy {
 public:
  CrossEntropy() = default;

  ~CrossEntropy() = default;

  MultidimensionalArrays Forward(const MultidimensionalArrays &input,
                                 const Index *labels);
};

}  // namespace Learning
}  // namespace KD
#endif