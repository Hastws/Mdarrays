#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_FUNCTION_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_FUNCTION_H

#include <cstring>
#include <memory>
#include <type_traits>

#include "exp/exp.h"
#include "exp/exp_impl.h"
#include "exp/operator/basic_op.h"
#include "exp/operator/conv.h"
#include "exp/operator/log_softmax.h"
#include "exp/operator/softmax.h"
#include "exp/operator/matrix_op.h"
#include "exp/operator/nll_loss.h"
#include "exp/operator/reduce_op.h"
#include "memory_pool/allocator.h"
#include "utils/exception.h"

namespace Autoalg {

template <typename ImplType>
struct Exp;

namespace Operator {

template <typename OperatorType, typename OIType>
// OIType = OperandImplType
Exp<UnaryExpImpl<OperatorType, OIType>> CreateUnaryOperationFunction(
    const Exp<OIType> &operand) {
  return Exp<UnaryExpImpl<OperatorType, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<OperatorType, OIType>>(
          operand.ImplPtr()));
}

template <typename OperatorType, typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<OperatorType, LhsImplType, RhsImplType>>
CreateBinaryOperationFunction(const Exp<LhsImplType> &lhs,
                              const Exp<RhsImplType> &rhs) {
  return Exp<BinaryExpImpl<OperatorType, LhsImplType, RhsImplType>>(
      Allocator::UniqueConstruct<
          BinaryExpImpl<OperatorType, LhsImplType, RhsImplType>>(
          lhs.ImplPtr(), rhs.ImplPtr()));
}

template <typename LhsImplType, typename RhsImplType>
typename std::enable_if<LhsImplType::Operator::Grad::AllowBroadcast::value &&
                            RhsImplType::Operator::Grad::AllowBroadcast::value,
                        void>::type
CheckBroadcast(const LhsImplType &lhs, const RhsImplType &rhs) {
  CHECK_EXP_BROADCAST(lhs, rhs);
}

template <typename LhsImplType, typename RhsImplType>
typename std::enable_if<!(LhsImplType::Operator::Grad::AllowBroadcast::value &&
                          RhsImplType::Operator::Grad::AllowBroadcast::value),
                        void>::type
CheckBroadcast(const LhsImplType &lhs, const RhsImplType &rhs) {
  CHECK_EXP_SAME_SHAPE(lhs, rhs);
}

// function for basic operation
inline Exp<UnaryExpImpl<Constant, BasicData>> CreateOperationConstant(
    BasicData value, IndexArray &&Size) {
  return Exp<UnaryExpImpl<Constant, BasicData>>(
      Allocator::UniqueConstruct<UnaryExpImpl<Constant, BasicData>>(
          value, std::move(Size)));
}

inline Exp<UnaryExpImpl<Constant, BasicData>> CreateOperationConstant(
    BasicData value, const IndexArray &Size) {
  return Exp<UnaryExpImpl<Constant, BasicData>>(
      Allocator::UniqueConstruct<UnaryExpImpl<Constant, BasicData>>(value,
                                                                    Size));
}

template <typename OIType>
Exp<UnaryExpImpl<Minus, OIType>> CreateOperationMinus(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Minus, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Minus, OIType>> operator-(const Exp<OIType> &operand) {
  return CreateOperationMinus<OIType>(operand);
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>> CreateOperationAdd(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs) {
  CheckBroadcast(lhs.Impl(), rhs.Impl());
  return CreateBinaryOperationFunction<Add, LhsImplType, RhsImplType>(lhs, rhs);
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>> operator+(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs) {
  return CreateOperationAdd<LhsImplType, RhsImplType>(lhs, rhs);
}

template <typename LhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, UnaryExpImpl<Constant, BasicData>>>
operator+(const Exp<LhsImplType> &lhs, BasicData data) {
  auto &lhs_impl = lhs.Impl();
  return lhs + CreateOperationConstant(
                   data, static_cast<const IndexArray &>(lhs_impl.Size()));
}

template <typename RhsImplType>
Exp<BinaryExpImpl<Add, UnaryExpImpl<Constant, BasicData>, RhsImplType>>
operator+(BasicData data, const Exp<RhsImplType> &rhs) {
  auto &lhs_impl = rhs.Impl();
  return CreateOperationConstant(
             data, static_cast<const IndexArray &>(lhs_impl.Size())) +
         rhs;
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>> CreateOperationMul(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs) {
  CheckBroadcast(lhs.Impl(), rhs.Impl());
  return CreateBinaryOperationFunction<Mul, LhsImplType, RhsImplType>(lhs, rhs);
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>> operator*(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs) {
  return CreateOperationMul<LhsImplType, RhsImplType>(lhs, rhs);
}

template <typename LhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, UnaryExpImpl<Constant, BasicData>>>
operator*(const Exp<LhsImplType> &lhs, BasicData data) {
  auto &lhs_impl = lhs.Impl();
  return lhs * CreateOperationConstant(
                   data, static_cast<const IndexArray &>(lhs_impl.Size()));
}

template <typename RhsImplType>
Exp<BinaryExpImpl<Mul, UnaryExpImpl<Constant, BasicData>, RhsImplType>>
operator*(BasicData data, const Exp<RhsImplType> &rhs) {
  auto &lhs_impl = rhs.Impl();
  return CreateOperationConstant(
             data, static_cast<const IndexArray &>(lhs_impl.Size())) *
         rhs;
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>> CreateOperationSub(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs) {
  CheckBroadcast(lhs.Impl(), rhs.Impl());
  return CreateBinaryOperationFunction<Sub, LhsImplType, RhsImplType>(lhs, rhs);
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>> operator-(
    const Exp<LhsImplType> &lhs, const Exp<RhsImplType> &rhs) {
  return CreateOperationSub<LhsImplType, RhsImplType>(lhs, rhs);
}

template <typename LhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, UnaryExpImpl<Constant, BasicData>>>
operator-(const Exp<LhsImplType> &lhs, BasicData data) {
  auto &lhs_impl = lhs.Impl();
  return lhs - CreateOperationConstant(
                   data, static_cast<const IndexArray &>(lhs_impl.Size()));
}

template <typename RhsImplType>
Exp<BinaryExpImpl<Sub, UnaryExpImpl<Constant, BasicData>, RhsImplType>>
operator-(BasicData data, const Exp<RhsImplType> &rhs) {
  auto &lhs_impl = rhs.Impl();
  return CreateOperationConstant(
             data, static_cast<const IndexArray &>(lhs_impl.Size())) -
         rhs;
}

template <typename OIType>
Exp<UnaryExpImpl<ReLU, OIType>> CreateOperationRelu(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<ReLU, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Sigmoid, OIType>> CreateOperationSigmoid(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Sigmoid, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Tanh, OIType>> CreateOperationTanh(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Tanh, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<ExpFunction, OIType>> CreateOperationExp(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<ExpFunction, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Log, OIType>> CreateOperationLog(const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Log, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Sqrt, OIType>> CreateOperationSqrt(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Sqrt, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Rsqrt, OIType>> CreateOperationRsqrt(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Rsqrt, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Reciprocal, OIType>> CreateOperationReciprocal(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Reciprocal, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Softplus, OIType>> CreateOperationSoftplus(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Softplus, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Swish, OIType>> CreateOperationSwish(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Swish, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Mish, OIType>> CreateOperationMish(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Mish, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<GELU, OIType>> CreateOperationGelu(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<GELU, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<HardSigmoid, OIType>> CreateOperationHardSigmoid(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<HardSigmoid, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<HardSwish, OIType>> CreateOperationHardSwish(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<HardSwish, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Abs, OIType>> CreateOperationAbs(const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Abs, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<LeakyReLU, OIType>> CreateOperationLeakReLU(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<LeakyReLU, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<ELU, OIType>> CreateOperationLeakELU(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<ELU, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<ReLU6, OIType>> CreateOperationLeakReLU6(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<ReLU6, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Log1p, OIType>> CreateOperationLeakLog1p(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Log1p, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<Expm1, OIType>> CreateOperationExpm1(
    const Exp<OIType> &operand) {
  return CreateUnaryOperationFunction<Expm1, OIType>(operand);
}

// function for matrix operation
template <typename OIType>
Exp<UnaryExpImpl<MatrixTranspose, OIType>> CreateOperationMatrixTranspose(
    const Exp<OIType> &operand) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 2,
              "Matrix Transpose is only supported for 2D "
              "Mdarray, but got "
                  << operand.Impl().DimensionsSize() << " one");
  return CreateUnaryOperationFunction<MatrixTranspose, OIType>(operand);
}

template <typename OIType>
Exp<UnaryExpImpl<BatchMatrixTranspose, OIType>>
CreateOperationBatchMatrixTranspose(const Exp<OIType> &operand) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 3,
              "Batch Matrix Transpose is only supported for 3D "
              "Mdarray, but got "
                  << operand.Impl().DimensionsSize() << " one");
  return CreateUnaryOperationFunction<BatchMatrixTranspose, OIType>(operand);
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<MatrixMul, LhsImplType, RhsImplType>>
CreateOperationMatrixMul(const Exp<LhsImplType> &lhs,
                         const Exp<RhsImplType> &rhs) {
#ifndef NDEBUG
  auto &lhs_impl = lhs.Impl();
  auto &rhs_impl = rhs.Impl();
  CHECK_TRUE(lhs_impl.DimensionsSize() == 2 && rhs_impl.DimensionsSize() == 2,
             "Matrices expected, got " << lhs_impl.DimensionsSize() << " and "
                                       << rhs_impl.DimensionsSize()
                                       << " Mdarray.");
  CHECK_EQUAL(lhs_impl.Size(1), rhs_impl.Size(0),
              "Size mismatch, m1: ["
                  << lhs_impl.Size(0) << ", " << lhs_impl.Size(1) << "], m2: ["
                  << rhs_impl.Size(0) << ", " << rhs_impl.Size(1) << "].");
#endif
  return CreateBinaryOperationFunction<MatrixMul, LhsImplType, RhsImplType>(
      lhs, rhs);
}

template <typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<BatchMatrixMul, LhsImplType, RhsImplType>>
CreateOperationBatchMatrixMul(const Exp<LhsImplType> &lhs,
                              const Exp<RhsImplType> &rhs) {
#ifndef NDEBUG
  auto &lhs_impl = lhs.Impl();
  auto &rhs_impl = rhs.Impl();
  CHECK_TRUE(lhs_impl.DimensionsSize() == 3 && rhs_impl.DimensionsSize() == 3,
             "Baths of Matrices expected, got "
                 << lhs_impl.DimensionsSize() << " and "
                 << rhs_impl.DimensionsSize() << " Mdarray");
  CHECK_TRUE(lhs_impl.Size(0) == rhs_impl.Size(0),
             "Bath sizes, " << lhs_impl.Size(0) << " and " << rhs_impl.Size(0)
                            << ", doesn't match.");
  CHECK_EQUAL(lhs_impl.Size(2), rhs_impl.Size(1),
              "Size mismatch, m1: ["
                  << lhs_impl.Size(1) << ", " << lhs_impl.Size(2) << "], m2: ["
                  << rhs_impl.Size(1) << ", " << rhs_impl.Size(2) << "].");
#endif
  return CreateBinaryOperationFunction<BatchMatrixMul, LhsImplType,
                                       RhsImplType>(lhs, rhs);
}

// function for CreateOperationLogSoftmax
template <typename OIType>
Exp<UnaryExpImpl<LogSoftmax, OIType>> CreateOperationLogSoftmax(
    const Exp<OIType> &operand) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 2,
              "CreateOperationLogSoftmax Only supported for 2D "
              "Mdarray, but got a "
                  << operand.Impl().DimensionsSize() << " one");
  return CreateUnaryOperationFunction<LogSoftmax, OIType>(operand);
}
// function for CreateOperationSoftMax
template <typename OIType>
Exp<UnaryExpImpl<Softmax, OIType>> CreateOperationSoftmax(
    const Exp<OIType> &operand) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 2,
              "CreateOperationLogSoftmax Only supported for 2D "
              "Mdarray, but got a "
                  << operand.Impl().DimensionsSize() << " one");
  return CreateUnaryOperationFunction<Softmax, OIType>(operand);
}

// function for reduce operator
template <typename OIType>
Exp<UnaryExpImpl<Mean, OIType>> CreateOperationMean(const Exp<OIType> &operand,
                                                    Index dim) {
  CHECK_IN_RANGE(dim, 0, operand.Impl().DimensionsSize(),
                 "Dimension out of range (expected to be in range of [0, "
                     << operand.Impl().DimensionsSize() << "), but got " << dim
                     << ")");
  return Exp<UnaryExpImpl<Mean, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<Mean, OIType>>(operand.ImplPtr(),
                                                             dim));
}

template <typename OIType>
Exp<UnaryExpImpl<Max, OIType>> CreateOperationMax(const Exp<OIType> &operand,
                                                  Index dim) {
  CHECK_IN_RANGE(dim, 0, operand.Impl().DimensionsSize(),
                 "Dimension out of range (expected to be in range of [0, "
                     << operand.Impl().DimensionsSize() << "), but got " << dim
                     << ")");
  return Exp<UnaryExpImpl<Max, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<Max, OIType>>(operand.ImplPtr(),
                                                            dim));
}

template <typename OIType>
Exp<UnaryExpImpl<Argmax, OIType>> CreateOperationArgmax(
    const Exp<OIType> &operand, Index dim) {
  CHECK_IN_RANGE(dim, 0, operand.Impl().DimensionsSize(),
                 "Dimension out of range (expected to be in range of [0, "
                     << operand.Impl().DimensionsSize() << "), but got " << dim
                     << ")");
  return Exp<UnaryExpImpl<Argmax, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<Argmax, OIType>>(
          operand.ImplPtr(), dim));
}

// function for CreateOperationNllLoss
template <typename OIType>
Exp<UnaryExpImpl<NLLLoss, OIType>> CreateOperationNllLoss(
    const Exp<OIType> &operand, const std::shared_ptr<Index> &labels_ptr,
    Index n_label = -1) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 2,
              "NLL Loss is only supported for 2D Mdarray, but got "
                  << operand.Impl().DimensionsSize() << " one.");
#ifndef NDEBUG
  Index n_batch = operand.Impl().Size(0);
  Index n_cls = operand.Impl().Size(1);
  CHECK_TRUE(
      n_label == Autoalg::Index(-1) || n_label == n_batch,
      "Batch Size mismatch, x: " << n_batch << ", labels: " << n_label << ".");

  auto labels = labels_ptr.get();
  for (Index i = 0; i < n_batch; ++i) {
    CHECK_IN_RANGE(labels[i], 0, n_cls,
                   n_cls << " classes got label of " << labels[i]);
  }
#endif
  return Exp<UnaryExpImpl<NLLLoss, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<NLLLoss, OIType>>(
          operand.ImplPtr(), labels_ptr));
}

template <typename OIType>
Exp<UnaryExpImpl<NLLLoss, OIType>> CreateOperationNllLoss(
    const Exp<OIType> &operand, const Index *labels, Index n_label = -1) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 2,
              "NLL Loss is only supported for 2D Mdarray, but got "
                  << operand.Impl().DimensionsSize() << " one.");

  Index n_batch = operand.Impl().Size(0);
#ifndef NDEBUG
  Index n_cls = operand.Impl().Size(1);
  CHECK_TRUE(n_label == Autoalg::Index(-1) || n_label == n_batch,
             "Batch Size mismatch, x: " << n_batch << ", labels: " << n_label);

  for (Index i = 0; i < n_batch; ++i)
    CHECK_IN_RANGE(labels[i], 0, n_cls,
                   n_cls << " classes got label of " << labels[i]);
#endif

  std::shared_ptr<Index> labels_ptr =
      Allocator::SharedAllocate<Index>(n_batch * sizeof(Index));
  std::memcpy(labels_ptr.get(), labels, n_batch * sizeof(Index));

  return Exp<UnaryExpImpl<NLLLoss, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<NLLLoss, OIType>>(
          operand.ImplPtr(), labels_ptr));
}

// function for conv
template <typename OIType>
Exp<UnaryExpImpl<Img2col, OIType>> CreateOperationImgToCol(
    const Exp<OIType> &operand, const Img2col::MatrixSize &kernel_size,
    const Img2col::MatrixSize &stride_size,
    const Img2col::MatrixSize &padding_size) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 4,
              "Img2col is only supported for 4D Mdarray, but got a "
                  << operand.Impl().DimensionsSize() << " one");
  CHECK_INDEX_VALID(kernel_size.first, "Invalid kernel_size.");
  CHECK_INDEX_VALID(kernel_size.second, "Invalid kernel_size.");
  CHECK_IN_RANGE(stride_size.first, 1, INDEX_MAX, "Invalid stride_size.");
  CHECK_IN_RANGE(stride_size.second, 1, INDEX_MAX, "Invalid stride_size.");
  CHECK_INDEX_VALID(padding_size.first, "Invalid padding_size.");
  CHECK_INDEX_VALID(padding_size.second, "Invalid padding_size.");
  CHECK_INDEX_VALID(
      operand.Impl().Size(2) + 2 * padding_size.first - kernel_size.first,
      "Kernel Size (" << kernel_size.first << ", " << kernel_size.second
                      << ") is too large");
  CHECK_INDEX_VALID(
      operand.Impl().Size(3) + 2 * padding_size.second - kernel_size.second,
      "Kernel Size (" << kernel_size.first << ", " << kernel_size.second
                      << ") is too large.");
  return Exp<UnaryExpImpl<Img2col, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<Img2col, OIType>>(
          operand.ImplPtr(), kernel_size, stride_size, padding_size));
}

template <typename OIType>
Exp<UnaryExpImpl<MaxPool2d, OIType>> CreateOperationMaxPool2d(
    const Exp<OIType> &operand, const MaxPool2d::MatrixSize &kernel_size,
    const MaxPool2d::MatrixSize &stride_size,
    const MaxPool2d::MatrixSize &padding_size) {
  CHECK_EQUAL(operand.Impl().DimensionsSize(), 4,
              "MaxPool2d is only supported for 4D Mdarray, but got a "
                  << operand.Impl().DimensionsSize() << " one");
  CHECK_INDEX_VALID(kernel_size.first, "Invalid kernel_size.");
  CHECK_INDEX_VALID(kernel_size.second, "Invalid kernel_size.");
  CHECK_IN_RANGE(stride_size.first, 1, INDEX_MAX, "Invalid stride_size.");
  CHECK_IN_RANGE(stride_size.second, 1, INDEX_MAX, "Invalid stride_size.");
  CHECK_INDEX_VALID(padding_size.first, "Invalid padding_size.");
  CHECK_INDEX_VALID(padding_size.second, "Invalid padding_size.");
  CHECK_INDEX_VALID(
      operand.Impl().Size(2) + 2 * padding_size.first - kernel_size.first,
      "Kernel Size (" << kernel_size.first << ", " << kernel_size.second
                      << ") is too large.");
  CHECK_INDEX_VALID(
      operand.Impl().Size(3) + 2 * padding_size.second - kernel_size.second,
      "Kernel Size (" << kernel_size.first << ", " << kernel_size.second
                      << ") is too large.");
  return Exp<UnaryExpImpl<MaxPool2d, OIType>>(
      Allocator::UniqueConstruct<UnaryExpImpl<MaxPool2d, OIType>>(
          operand.ImplPtr(), kernel_size, stride_size, padding_size));
}
}  // namespace Operator
}  // namespace Autoalg

#endif