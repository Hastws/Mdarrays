#include "learning/module.h"

#include <utility>

#include "exp/function.h"
#include "learning/init.h"

namespace KD {
namespace Learning {
Linear::Linear(Index in_features, Index out_features)
    : weight_(Shape{out_features, in_features}, true),
      bias_(Shape{1, out_features}, true) {
  KaimingInitializer weight_init(weight_);
  weight_init.Init();
  UniformInitializer bias_init(bias_);
  bias_init.Init();
}

MultidimensionalArrays Linear::Forward(const MultidimensionalArrays &x) {
  MultidimensionalArrays y1 = Operator::CreateOperationMatrixMul(
      x, Operator::CreateOperationMatrixTranspose(weight_));
  MultidimensionalArrays y2 = y1 + bias_;
  return y2;
}

ParamsDict Linear::Parameters() {
  return {{"weight", weight_}, {"bias", bias_}};
}

LinearWithReLU::LinearWithReLU(Index in_features, Index out_features)
    : Linear(in_features, out_features) {}

MultidimensionalArrays LinearWithReLU::Forward(
    const MultidimensionalArrays &x) {
  MultidimensionalArrays y1 = Operator::CreateOperationMatrixMul(
      x, Operator::CreateOperationMatrixTranspose(weight_));
  MultidimensionalArrays y2 = Operator::CreateOperationRelu(y1 + bias_);
  return y2;
}
}  // namespace Learning
}  // namespace KD

namespace KD {
namespace Learning {
Conv2d::Conv2d(Index in_channels, Index out_channels, MatrixSize kernel_size,
               MatrixSize stride, MatrixSize padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      weight_(Shape{out_channels_,
                    in_channels_ * kernel_size_.first * kernel_size_.second},
              true) {
  KaimingInitializer weight_init(weight_);
  weight_init.Init();
}

MultidimensionalArrays Conv2d::Forward(const MultidimensionalArrays &x) {
  auto col_exp =
      Operator::CreateOperationImgToCol(x, kernel_size_, stride_, padding_);

  MultidimensionalArrays y1 = Operator::CreateOperationMatrixMul(
      MultidimensionalArrays(col_exp),
      Operator::CreateOperationMatrixTranspose(weight_));

  auto &&conv_feat_size = col_exp.Impl().ConvFeatSize();
  MultidimensionalArrays y2 = y1.View(
      {conv_feat_size.first, conv_feat_size.second, x.Size(0), out_channels_});
  MultidimensionalArrays y3 = y2.Permute({2, 3, 0, 1});
  return y3;
}

ParamsDict Conv2d::Parameters() { return {{"weight", weight_}}; }

Conv2dWithReLU::Conv2dWithReLU(Index in_channels, Index out_channels,
                               const MatrixSize &kernel_size,
                               const MatrixSize &stride,
                               const MatrixSize &padding)
    : Conv2d(in_channels, out_channels, kernel_size, stride, padding) {}

MultidimensionalArrays Conv2dWithReLU::Forward(
    const MultidimensionalArrays &x) {
  auto col_exp =
      Operator::CreateOperationImgToCol(x, kernel_size_, stride_, padding_);

  MultidimensionalArrays y1 =
      Operator::CreateOperationRelu(Operator::CreateOperationMatrixMul(
          MultidimensionalArrays(col_exp),
          Operator::CreateOperationMatrixTranspose(weight_)));

  auto &conv_feat_size = col_exp.Impl().ConvFeatSize();
  MultidimensionalArrays y2 = y1.View(
      {conv_feat_size.first, conv_feat_size.second, x.Size(0), out_channels_});
  MultidimensionalArrays y3 = y2.Permute({2, 3, 0, 1});
  return y3;
}
}  // namespace Learning
}  // namespace KD

namespace KD {
namespace Learning {

MaxPool2d::MaxPool2d(MatrixSize kernel_size, MatrixSize stride,
                     MatrixSize padding)
    : kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)) {}

MultidimensionalArrays MaxPool2d::Forward(const MultidimensionalArrays &x) {
  auto col_exp =
      Operator::CreateOperationImgToCol(x, kernel_size_, stride_, padding_);
  MultidimensionalArrays y1 = col_exp;

  auto &conv_feat_size = col_exp.Impl().ConvFeatSize();
  MultidimensionalArrays y2 =
      y1.View({conv_feat_size.first, conv_feat_size.second, x.Size(0),
               x.Size(1), kernel_size_.first * kernel_size_.second});
  MultidimensionalArrays y3 = Operator::CreateOperationMax(y2, 4);
  MultidimensionalArrays y4 = y3.Permute({2, 3, 0, 1});
  return y4;
}

ParamsDict MaxPool2d::Parameters() { return {}; }

MultidimensionalArrays CrossEntropy::Forward(
    const MultidimensionalArrays &input, const Index *labels) {
  auto log_its = Operator::CreateOperationLogSoftmax(input);
  auto nll = Operator::CreateOperationNllLoss(log_its, labels);
  MultidimensionalArrays loss = Operator::CreateOperationMean(nll, 0);
  return loss;
}
}  // namespace Learning
}  // namespace KD