#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_CONV_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_EXP_OPERATOR_CONV_H

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <utility>

#include "utils/array.h"
#include "utils/base_config.h"
namespace KD {
namespace Operator {
// Acceleration of convolution operation
struct Img2col {
  using MatrixSize = std::pair<Index, Index>;

  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 2;
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &, const MatrixSize &Size) {
    return idx == 0 ? Size.first : Size.second;
  }

  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       const MatrixSize &kernel_size,
                       const MatrixSize &stride_size,
                       const MatrixSize &padding_size,
                       const MatrixSize &out_size) {
    Index n_batch = operand.Size(0);
    Index h = operand.Size(2);
    Index w = operand.Size(3);
    Index col = indexes[0];
    Index row = indexes[1];

    // Size(0) = oh * ow * b
    Index h_idx = col / (out_size.second * n_batch);
    col %= (n_batch * out_size.second);
    Index w_idx = col / n_batch;
    Index b_idx = col % n_batch;

    // Size(1) = c * kh * kw
    Index c_idx = row / (kernel_size.first * kernel_size.second);
    row %= kernel_size.first * kernel_size.second;
    Index kh_idx = row / kernel_size.second;
    Index kw_idx = row % kernel_size.second;

    // In fact, Index is unsigned int, which can't be negative.
    // So we can't subtract padding_size here.
    h_idx = h_idx * stride_size.first + kh_idx;
    w_idx = w_idx * stride_size.second + kw_idx;

    if (h_idx < padding_size.first || h_idx >= h + padding_size.first ||
        w_idx < padding_size.second || w_idx >= w + padding_size.second) {
      return 0;
    }

    h_idx -= padding_size.first;
    w_idx -= padding_size.second;
    IndexArray operand_indexes{b_idx, c_idx, h_idx, w_idx};

    return operand.Eval(operand_indexes);
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    template <typename GradType, typename OperandType>
    static BasicData Map(IndexArray &indexes, const GradType &grad,
                         const OperandType &operand,
                         const MatrixSize &kernel_size,
                         const MatrixSize &stride_size,
                         const MatrixSize &padding_size,
                         const MatrixSize &out_size) {
      // operand Size: (b, c, h, w)
      // Grad Size: (oh*ow*b, c*kh*kw)
      Index n_batch = operand.Size(0);
      Index img_h = operand.Size(2) + (padding_size.first << 1);
      Index img_w = operand.Size(3) + (padding_size.second << 1);
      Index kh_idx, kw_idx;  // location in a patch
      Index ph_idx, pw_idx;  // location of the left top point of a patch
      IndexArray grad_indexes(2);
      BasicData total_grad = 0;

      Index c_step = kernel_size.first * kernel_size.second;
      Index kh_step = kernel_size.second;
      Index oh_step = out_size.second * n_batch;
      Index ow_step = n_batch;

      indexes[2] += padding_size.first;
      indexes[3] += padding_size.second;
      for (kh_idx = 0; kh_idx < kernel_size.first && kh_idx <= indexes[2];
           ++kh_idx) {
        for (kw_idx = 0; kw_idx < kernel_size.second && kw_idx <= indexes[3];
             ++kw_idx) {
          ph_idx = indexes[2] - kh_idx;
          pw_idx = indexes[3] - kw_idx;

          if (ph_idx + kernel_size.first > img_h ||
              pw_idx + kernel_size.second > img_w ||
              ph_idx % stride_size.first || pw_idx % stride_size.second) {
            continue;
          }

          grad_indexes[0] = ph_idx / stride_size.first * oh_step +
                            pw_idx / stride_size.second * ow_step + indexes[0];
          grad_indexes[1] = indexes[1] * c_step + kh_idx * kh_step + kw_idx;
          total_grad += grad.Eval(grad_indexes);
        }
      }
      return total_grad;
    }
  };
};

struct MaxPool2d {
  using MatrixSize = std::pair<Index, Index>;

  template <typename OperandType>
  static Index DimensionsSize(const OperandType &) {
    return 4;
  }

  template <typename OperandType>
  static Index Size(Index idx, const OperandType &operand,
                    const MatrixSize &out_size) {
    switch (idx) {
      case 1:
        return operand.Size(1);  // num_channel
      case 2:
        return out_size.first;
      case 3:
        return out_size.second;
      default:
        return operand.Size(0);  // num_batch
    }
  }

  template <typename OperandType>
  static BasicData Map(IndexArray &indexes, const OperandType &operand,
                       const MatrixSize &kernel_size,
                       const MatrixSize &stride_size,
                       const MatrixSize &padding_size) {
    Index h = operand.Size(2);
    Index w = operand.Size(3);
    Index h_start = indexes[2] * stride_size.first;
    Index w_start = indexes[3] * stride_size.second;
    Index h_end = h_start + kernel_size.first;
    Index w_end = w_start + kernel_size.second;
    IndexArray operand_indexes(indexes);

    BasicData value, max_value = DATA_MIN;
    for (Index i = h_start; i < h_end; ++i) {
      if (i < padding_size.first || i >= h + padding_size.first) {
        max_value = std::max(max_value, 0.);
        continue;
      }
      for (Index j = w_start; j < w_end; ++j) {
        if (j < padding_size.second || j >= w + padding_size.second) {
          value = 0;
        } else {
          operand_indexes[2] = i - padding_size.first;
          operand_indexes[3] = j - padding_size.second;
          value = operand.Eval(operand_indexes);
        }
        max_value = std::max(max_value, value);
      }
    }
    return max_value;
  }

  struct Grad {
    using AllowBroadcast = std::false_type;
    using IsLhs = std::false_type;
    using IsRhs = std::false_type;

    static BasicData Map(IndexArray &, BasicData) {
      THROW_ERROR("Not implement error");
    }
  };
};

}  // namespace Operator
}  // namespace KD

#endif