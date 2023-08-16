#include <chrono>

#include "exp/function.h"
#include "learning/init.h"
#include "learning/module.h"
#include "learning/optimizer.h"
#include "multidimensional_arrays/multidimensional_arrays.h"
#include "multidimensional_arrays/shape.h"
#include "multidimensional_arrays/storage.h"
#include "utils/exception.h"

void TestMultidimensionalArrays();

void TestBasicOperator();

void TestMatrixOperator();

void TestNumericOperator();

void TestConvOperator();

void TestMultidimensionalArraysBackward();

void TestBasicOperatorBackward();

void TestMatrixOperatorBackward();

void TestNumericOperatorBackward();

void TestImg2colOperatorBackward();

void TestBroadcastingOperatorBackward();

void TestConv2dModule();

void TestLinearModule();

void TestMaxPool2dModule();

void TestCrossEntropyModule();

void TestOptimizer();

int main() {
  std::chrono::steady_clock::time_point start_tp =
      std::chrono::steady_clock::now();

  TestMultidimensionalArrays();
  TestBasicOperator();
  TestMatrixOperator();
  TestNumericOperator();
  TestConvOperator();
  TestMultidimensionalArraysBackward();
  TestBasicOperatorBackward();
  TestMatrixOperatorBackward();
  TestNumericOperatorBackward();
  TestImg2colOperatorBackward();
  TestBroadcastingOperatorBackward();
  TestConv2dModule();
  TestLinearModule();
  TestMaxPool2dModule();
  TestCrossEntropyModule();
  TestOptimizer();

  std::chrono::steady_clock::time_point end_tp =
      std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_tp -
                                                                start_tp);
  LOG_INFO("Test success, test took :[" << time_span.count() << "]s.")

  return 0;
}

void TestMultidimensionalArrays() {
  KD::BasicData data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::MultidimensionalArrays mda_1(data, KD::Shape({3, 4}));
  for (KD::Index i = 0, idx = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value = mda_1[{i, j}];
      CHECK_FLOAT_EQUAL(value, data[idx], "Check basic index.");
      idx++;
    }
  }
  // [[1, 2, 3, 4],
  // [5, 6, 7, 8],
  // [9, 10, 11, 12]]
  LOG_INFO(mda_1)

  auto mda_2 = mda_1.Transpose(0, 1);
  for (KD::Index i = 0; i < 4; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = mda_1[{j, i}];
      KD::BasicData value2 = mda_2[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "Check transpose.");
    }
  }
  // [[1, 5, 9],
  // [2, 6, 10],
  // [3, 7, 11],
  // [4, 8, 12]]
  LOG_INFO(mda_2)

  auto mda_3 = mda_2.Slice(1, 1, 3);
  auto shape_t3 = mda_3.Size();
  CHECK_TRUE(shape_t3[0] == 4 && shape_t3[1] == 2, "Check slice.");
  for (KD::Index i = 0; i < 4; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      KD::BasicData value1 = mda_2[{i, j + 1}];
      KD::BasicData value2 = mda_3[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "Check slice.");
    }
  }

  // [[5, 9],
  // [6, 10],
  // [7, 11],
  // [8, 12]]
  LOG_INFO(mda_3)

  auto mda_4 = mda_1.View({3, 2, 2});
  auto shape_t4 = mda_4.Size();
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 2; ++k) {
        KD::BasicData value1 = mda_1[{i, j * 2 + k}];
        KD::BasicData value2 = mda_4[{i, j, k}];
        CHECK_FLOAT_EQUAL(value1, value2, "Check view.");
      }
    }
  }
  // [[[1, 2],
  // [3, 4]],
  // [[5, 6],
  // [7, 8]],
  // [[9, 10],
  // [11, 12]]]
  LOG_INFO(mda_4)

  auto mda_5 = mda_4.Unsqueeze(0).Unsqueeze(2);
  CHECK_EQUAL(mda_5.DimensionsSize(), 5, "Check unsqueeze.");
  KD::Shape shape_t5({1, 3, 1, 2, 2});
  for (KD::Index i = 0; i < 5; ++i)
    CHECK_EQUAL(mda_5.Size(i), shape_t5[i], "Check unsqueeze.");
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 2; ++k) {
        KD::BasicData value1 = mda_4[{i, j, k}];
        KD::BasicData value2 = mda_5[{0, i, 0, j, k}];
        CHECK_FLOAT_EQUAL(value1, value2, "Check unsqueeze.");
      }
    }
  }
  // [[
  // [[[1, 2],[3, 4]]],
  // [[[5, 6],[7, 8]]],
  // [[[9, 10],[11, 12]]]
  // ]]
  LOG_INFO(mda_5)

  auto mda_6 = mda_5.Squeeze();
  CHECK_EQUAL(mda_6.DimensionsSize(), mda_4.DimensionsSize(), "Check squeeze.");
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 2; ++k) {
        KD::BasicData value1 = mda_4[{i, j, k}];
        KD::BasicData value2 = mda_6[{i, j, k}];
        CHECK_FLOAT_EQUAL(value1, value2, "Check squeeze.");
      }
    }
  }
  // [[[1, 2],[3, 4]],
  // [[5, 6],[7, 8]],
  // [[9, 10],[11, 12]]]
  LOG_INFO(mda_6)

  auto mda_7 = mda_5.Permute({0, 2, 3, 4, 1});
  CHECK_EQUAL(mda_7.DimensionsSize(), 5, "Check permute.");
  KD::Shape shape_t7({1, 1, 2, 2, 3});
  for (KD::Index i = 0; i < 5; ++i)
    CHECK_EQUAL(mda_7.Size(i), shape_t7[i], "Check permute.");
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 3; ++k) {
        KD::BasicData value1 = mda_7[{0, 0, i, j, k}];
        KD::BasicData value2 = mda_5[{0, k, 0, i, j}];
        CHECK_FLOAT_EQUAL(value1, value2, "Check permute.");
      }
    }
  }
  // [[
  // [
  // [[1, 5, 9],[2, 6, 10]],
  // [[3, 7, 11],[4, 8, 12]]
  // ]
  // ]]
  LOG_INFO(mda_7)
}

void TestBasicOperator() {
  KD::BasicData data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::MultidimensionalArrays mda_1(data, KD::Shape{3, 4}, true);
  KD::MultidimensionalArrays mda_2(data, KD::Shape{3, 4});

  KD::MultidimensionalArrays mda_1_1 = 5.0 + mda_1;
  std::cout << mda_1_1 << std::endl;
  mda_1_1.Backward();
  std::cout << mda_1.Grad() << std::endl;

  KD::MultidimensionalArrays mda_3 = mda_1 + mda_2;
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = mda_3[{i, j}];
      KD::BasicData value2 = 2 * mda_1[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check 1");
    }
  }

  // [[1, 2, 3, 4],
  // [5, 6, 7, 8],
  // [9, 10, 11, 12]]
  LOG_INFO(mda_1)
  // [[1, 2, 3, 4],
  // [5, 6, 7, 8],
  // [9, 10, 11, 12]]
  LOG_INFO(mda_2)
  // [[2, 4, 6, 8],
  // [10, 12, 14, 16],
  // [18, 20, 22, 24]]
  LOG_INFO(mda_3)

  mda_3 += mda_1;
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = mda_3[{i, j}];
      KD::BasicData value2 = 3 * mda_1[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check 2");
    }
  }
  // [[3, 6, 9, 12],
  // [15, 18, 21, 24],
  // [27, 30, 33, 36]]
  LOG_INFO(mda_3)

  KD::MultidimensionalArrays mda_4 = mda_1 * mda_2 + mda_3;
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = mda_4[{i, j}];
      KD::BasicData value2 = mda_1[{i, j}] * mda_2[{i, j}] + mda_3[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check 3");
    }
  }
  // [[4, 10, 18, 28],
  // [40, 54, 70, 88],
  // [108, 130, 154, 180]]
  LOG_INFO(mda_4)

  auto func = [&mda_1, &mda_2](const KD::MultidimensionalArrays &t3,
                               const KD::MultidimensionalArrays &t4) {
    auto add_exp = mda_1 + mda_2;
    auto mul_exp = -mda_1 * mda_2;
    return t3 * t4 - add_exp - mul_exp;
  };
  auto exp = func(mda_3, mda_4);
  // At this time, add_exp, mul_exp and other implicitly constructed Exp has
  // been deconstructed. But we expect the BinaryExpImpl hold by them is
  // still alive, untill the assignment of mda_5 completes.
  KD::MultidimensionalArrays mda_5 = exp;
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = mda_5[{i, j}];
      KD::BasicData value2 = mda_3[{i, j}] * mda_4[{i, j}] -
                             (mda_1[{i, j}] + mda_2[{i, j}]) -
                             (-mda_1[{i, j}] * mda_2[{i, j}]);
      CHECK_FLOAT_EQUAL(value1, value2, "check 3");
    }
  }
  // [[11, 60, 165, 344],
  // [615, 996, 1505, 2160],
  // [2979, 3980, 5181, 6600]]
  LOG_INFO(mda_5)

  KD::MultidimensionalArrays mda_6 = mda_1.View({2, 1, 1, 2, 3});
  KD::MultidimensionalArrays mda_7 = mda_1.View({2, 2, 1, 1, 3});
  // The shape (2, 2, 3) same as (2, 2, 3, 1, 1)
  KD::MultidimensionalArrays mda_8 = mda_1.View({2, 2, 3});
  // [[[[[1, 2, 3],[4, 5, 6]]]],
  // [[[[7, 8, 9],[10, 11, 12]]]]]
  LOG_INFO(mda_6)
  // [
  // [[[[1, 2, 3]]],[[[4, 5, 6]]]],
  // [[[[7, 8, 9]]],[[[10, 11, 12]]]]
  // ]
  LOG_INFO(mda_7)
  // [
  // [[1, 2, 3],[4, 5, 6]],
  // [[7, 8, 9],[10, 11, 12]]
  // ]
  LOG_INFO(mda_8)
  auto exp1 = mda_6 + mda_7;
  auto exp2 = -(mda_6 * mda_8);
  auto exp3 = mda_6 - mda_8;
  KD::MultidimensionalArrays mda_9 = exp1 + exp2 + exp3;
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 3; ++k) {
        for (KD::Index l = 0; l < 2; ++l) {
          for (KD::Index m = 0; m < 3; ++m) {
            KD::BasicData value1 = mda_9[{i, j, k, l, m}];
            KD::BasicData value2 =
                mda_6[{i, 0, 0, l, m}] + mda_7[{i, j, 0, 0, m}];
            value2 -= mda_6[{i, 0, 0, l, m}] * mda_8[{i, j, k}];
            value2 += mda_6[{i, 0, 0, l, m}] - mda_8[{i, j, k}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 3");
          }
        }
      }
    }
  }

  // (2, 2, 3, 2, 3)
  LOG_INFO(mda_9.Size())
  // [
  // [
  // [[[1, 3, 5],[4, 6, 8]],[[-1, 0, 1],[-1, 0, 1]],[[-3, -3, -3],[-6, -6,
  // -6]]],
  // [[[-2, -3, -4],[-8, -9, -10]],[[-4, -6, -8],[-13, -15, -17]],[[-6, -9,
  // -12],[-18, -21, -24]]]
  // ],
  // [
  // [[[-35, -39, -43],[-50, -54, -58]],[[-43, -48, -53],[-61, -66, -71]],[[-51,
  // -57, -63],[-72, -78, -84]]],
  // [[[-56, -63, -70],[-80, -87, -94]],[[-64, -72, -80],[-91, -99,
  // -107]],[[-72, -81, -90],[-102, -111, -120]]]
  // ]
  // ]
  LOG_INFO(mda_9)

  KD::MultidimensionalArrays mda_10 =
      mda_1.Transpose(0, 1) + KD::Operator::CreateOperationConstant(1, {4, 3});
  for (KD::Index i = 0; i < 4; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = mda_10[{i, j}];
      KD::BasicData value2 = mda_1[{j, i}] + 1;
      CHECK_FLOAT_EQUAL(value1, value2, "check5");
    }
  }
  // [[2, 6, 10],
  // [3, 7, 11],
  // [4, 8, 12],
  // [5, 9, 13]]
  LOG_INFO(mda_10)

  // assignment of uncontiguous multidimensional_arrays
  auto mda_11 = mda_1.Transpose(0, 1);
  KD::MultidimensionalArrays mda_12 = mda_2.Transpose(0, 1);
  KD::MultidimensionalArrays mda_13(data, mda_11.Size());
  mda_11 = mda_13;
  // Operation constant value is 0
  mda_12 = mda_11 + KD::Operator::CreateOperationConstant(0, {4, 3});
  for (KD::Index i = 0; i < 4; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = mda_11[{i, j}];
      KD::BasicData value2 = mda_12[{i, j}];
      KD::BasicData value3 = mda_13[{i, j}];
      CHECK_TRUE(value1 == value2 && value1 == value3, "check6");
    }
  }
  // [[1, 2, 3],
  // [4, 5, 6],
  // [7, 8, 9],
  // [10, 11, 12]]
  LOG_INFO(mda_11)
  // [[1, 2, 3],
  // [4, 5, 6],
  // [7, 8, 9],
  // [10, 11, 12]]
  LOG_INFO(mda_12)
  // [[1, 2, 3],
  // [4, 5, 6],
  // [7, 8, 9],
  // [10, 11, 12]]
  LOG_INFO(mda_13)
}

void TestMatrixOperator() {
  KD::BasicData data_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::BasicData data_2[] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121};
  KD::MultidimensionalArrays mda_1(data_1, KD::Shape{2, 6});
  KD::MultidimensionalArrays mda_2(data_2, KD::Shape{2, 6});
  // [[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]]
  LOG_INFO(mda_1)
  //[[11, 21, 31, 41, 51, 61],[71, 81, 91, 101, 111, 121]]
  LOG_INFO(mda_2)

  KD::MultidimensionalArrays mda_3 =
      KD::Operator::CreateOperationMatrixTranspose(
          KD::Operator::CreateOperationMatrixMul(mda_1, mda_2.Transpose(0, 1)));
  KD::BasicData t3_expect[2][2] = {{931, 2227}, {2191, 5647}};
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      KD::BasicData value1 = mda_3[{i, j}];
      KD::BasicData value2 = t3_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check 1");
    }
  }
  // [[931, 2227],[2191, 5647]]
  LOG_INFO(mda_3)

  KD::MultidimensionalArrays mda_4 = mda_1.View({3, 2, 2});
  KD::MultidimensionalArrays mda_5 = mda_2.View({3, 2, 2});
  // [[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9, 10],[11, 12]]]
  LOG_INFO(mda_4)
  // [[[11, 21],[31, 41]],[[51, 61],[71, 81]],[[91, 101],[111, 121]]]
  LOG_INFO(mda_5)
  KD::MultidimensionalArrays mda_6 =
      KD::Operator::CreateOperationBatchMatrixTranspose(
          KD::Operator::CreateOperationBatchMatrixMul(mda_4, mda_5));
  KD::BasicData t6_expect[3][2][2] = {{{73, 157}, {103, 227}},
                                      {{681, 925}, {791, 1075}},
                                      {{1929, 2333}, {2119, 2563}}};
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 2; ++k) {
        KD::BasicData value1 = mda_6[{i, j, k}];
        KD::BasicData value2 = t6_expect[i][j][k];
        CHECK_FLOAT_EQUAL(value1, value2, "check 2");
      }
    }
  }
  // [[[73, 157],[103, 227]],[[681, 925],[791, 1075]],[[1929, 2333],[2119,
  // 2563]]]
  LOG_INFO(mda_6)

  KD::MultidimensionalArrays mda_7 =
      KD::Operator::CreateOperationMatrixTranspose(mda_1);
  CHECK_EQUAL(mda_7.DimensionsSize(), 2, "check3");
  CHECK_EQUAL(mda_7.Size(0), 6, "check3");
  CHECK_EQUAL(mda_7.Size(1), 2, "check3");
  for (KD::Index i = 0; i < 6; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      KD::BasicData value1 = mda_1[{j, i}];
      KD::BasicData value2 = mda_7[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check 3");
    }
  }
  // [[1, 7],[2, 8],[3, 9],[4, 10],[5, 11],[6, 12]]
  LOG_INFO(mda_7)

  KD::MultidimensionalArrays mda_8(data_1, KD::Shape{2, 2, 3});
  // [[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]]
  LOG_INFO(mda_8)

  KD::MultidimensionalArrays mda_9 =
      KD::Operator::CreateOperationBatchMatrixTranspose(mda_8);
  CHECK_EQUAL(mda_9.DimensionsSize(), 3, "check4");
  CHECK_EQUAL(mda_9.Size(0), 2, "check4");
  CHECK_EQUAL(mda_9.Size(1), 3, "check4");
  CHECK_EQUAL(mda_9.Size(2), 2, "check4");
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      for (KD::Index k = 0; k < 2; ++k) {
        KD::BasicData value1 = mda_8[{i, k, j}];
        KD::BasicData value2 = mda_9[{i, j, k}];
        CHECK_FLOAT_EQUAL(value1, value2, "check 3");
      }
    }
  }
  // [[[1, 4],[2, 5],[3, 6]],[[7, 10],[8, 11],[9, 12]]]
  LOG_INFO(mda_9)
}

void TestNumericOperator() {
  KD::BasicData data_1[] = {0.585639, 0.612628, 0.241485, 0.097616,
                            0.035854, 0.723054, 0.131163, 0.884268,
                            0.193597, 0.694748, 0.650687, 0.738797};
  KD::MultidimensionalArrays mda_1(data_1, KD::Shape{3, 4});
  // [[0.585639, 0.612628, 0.241485, 0.097616],
  // [0.035854, 0.723054, 0.131163, 0.884268],
  // [0.193597, 0.694748, 0.650687, 0.738797]]
  LOG_INFO(mda_1)
  KD::BasicData t1_expect[3][4] = {
      {-1.208965, -1.181976, -1.553119, -1.696988},
      {-1.860054, -1.172853, -1.764744, -1.011639},
      {-1.784239, -1.283088, -1.327148, -1.239038}};
  KD::MultidimensionalArrays mda_2 =
      KD::Operator::CreateOperationLogSoftmax(mda_1);
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = mda_2[{i, j}];
      KD::BasicData value2 = t1_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
  }
  // [[-1.20896482874321, -1.18197582874321, -1.55311882874321,
  // -1.69698782874321],
  // [-1.86005323576525, -1.17285323576525, -1.76474423576525,
  // -1.01163923576525],
  // [-1.78423866041312, -1.28308766041312, -1.32714866041312,
  // -1.23903866041312]]
  LOG_INFO(mda_2)

  auto labels_ptr =
      KD::Allocator::SharedAllocate<KD::Index>(3 * sizeof(KD::Index));
  auto labels = labels_ptr.get();
  // The value of the corresponding position
  labels[0] = 2, labels[1] = 0, labels[2] = 3;
  KD::MultidimensionalArrays mda_3 =
      KD::Operator::CreateOperationNllLoss(mda_2, labels_ptr);
  CHECK_EQUAL(mda_3.DimensionsSize(), 1, "check2");
  CHECK_EQUAL(mda_3.Size(0), mda_2.Size(0), "check2");
  CHECK_FLOAT_EQUAL(mda_3[{0}], -t1_expect[0][2], "check2");
  CHECK_FLOAT_EQUAL(mda_3[{1}], -t1_expect[1][0], "check2");
  CHECK_FLOAT_EQUAL(mda_3[{2}], -t1_expect[2][3], "check2");
  // [1.55311882874321, 1.86005323576525, 1.23903866041312]
  LOG_INFO(mda_3)

  KD::BasicData data_2[] = {0.096237, -0.037000, 0.028076, 0.328307,
                            0.122271, -0.017293, 0.150791, 0.421008,
                            0.322066, -0.321352, 0.319534, -0.424081};
  KD::MultidimensionalArrays mda_4(data_2, KD::Shape{2, 2, 3});
  // [[[0.096237, -0.037, 0.028076],[0.328307, 0.122271, -0.017293]],[[0.150791,
  // 0.421008, 0.322066],[-0.321352, 0.319534, -0.424081]]]
  LOG_INFO(mda_4)

  KD::MultidimensionalArrays mda_5 = KD::Operator::CreateOperationMean(
      KD::Operator::CreateOperationSigmoid(
          KD::Operator::CreateOperationRelu(mda_4)),
      1);
  KD::BasicData t4_expect[][3] = {{0.552694, 0.515265, 0.503509},
                                  {0.518813, 0.591467, 0.539914}};
  CHECK_TRUE(
      mda_5.DimensionsSize() == 2 && mda_5.Size(0) == 2 && mda_5.Size(1) == 3,
      "check3");
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = mda_5[{i, j}];
      KD::BasicData value2 = t4_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check4");
    }
  }
  // [[0.552694042632973, 0.515264862011877,
  // 0.503509269484445],[0.518813240662988, 0.591467555207441,
  // 0.539913834749743]]
  LOG_INFO(mda_5)

  KD::MultidimensionalArrays mda_6 = KD::Operator::CreateOperationMean(
      KD::Operator::CreateOperationMean(mda_5, 0), 0);
  CHECK_TRUE(mda_6.DimensionsSize() == 1 && mda_6.Size(0) == 1, "check5");
  CHECK_FLOAT_EQUAL(mda_6.Item(), 0.536944, "check5");
  // [0.536943800791578]
  LOG_INFO(mda_6)

  KD::MultidimensionalArrays mda_7 =
      KD::Operator::CreateOperationArgmax(mda_4, 1);
  KD::Index t6_expect[][3] = {{1, 1, 0}, {0, 0, 0}};
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::Index value1 = static_cast<KD::Index>(mda_7[{i, j}]);
      KD::Index value2 = t6_expect[i][j];
      CHECK_EQUAL(value1, value2, "check6");
    }
  }
  // [[1, 1, 0],[0, 0, 0]]
  LOG_INFO(mda_7)

  KD::MultidimensionalArrays mda_8 = KD::Operator::CreateOperationMax(mda_4, 1);
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = mda_8[{i, j}];
      KD::BasicData value2 = mda_4[{i, t6_expect[i][j], j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check7");
    }
  }
  // [[0.328307, 0.122271, 0.028076],[0.150791, 0.421008, 0.322066]]
  LOG_INFO(mda_8)
}

void TestConvOperator() {
  KD::BasicData data[6][4] = {
      {0.4279, 0.7488, 0.3639, 0.5433}, {0.2849, 0.6536, 0.8932, 0.9341},
      {0.9640, 0.4822, 0.1887, 0.9457}, {0.2132, 0.0185, 0.0163, 0.9874},
      {0.2039, 0.8020, 0.3766, 0.6537}, {0.8543, 0.3589, 0.5178, 0.7816}};
  KD::MultidimensionalArrays dma_0(reinterpret_cast<KD::BasicData *>(data),
                                   KD::Shape{1, 1, 6, 4});
  // [[[[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]]]]
  LOG_INFO(dma_0)

  KD::MultidimensionalArrays dma_1 =
      KD::Operator::CreateOperationMaxPool2d(dma_0, {2, 2}, {1, 1}, {1, 1});
  KD::Index t1_size_expect[] = {1, 1, 7, 5};
  KD::BasicData t1_expect[7][5] = {{0.4279, 0.7488, 0.7488, 0.5433, 0.5433},
                                   {0.4279, 0.7488, 0.8932, 0.9341, 0.9341},
                                   {0.9640, 0.9640, 0.8932, 0.9457, 0.9457},
                                   {0.9640, 0.9640, 0.4822, 0.9874, 0.9874},
                                   {0.2132, 0.8020, 0.8020, 0.9874, 0.9874},
                                   {0.8543, 0.8543, 0.8020, 0.7816, 0.7816},
                                   {0.8543, 0.8543, 0.5178, 0.7816, 0.7816}};
  CHECK_EQUAL(dma_1.DimensionsSize(), 4, "check1");
  for (KD::Index i = 0; i < 4; ++i)
    CHECK_EQUAL(dma_1.Size(i), t1_size_expect[i], "check1");
  for (KD::Index i = 0; i < 7; ++i) {
    for (KD::Index j = 0; j < 5; ++j) {
      KD::BasicData value1 = dma_1[{0, 0, i, j}];
      KD::BasicData value2 = t1_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }
  // [[[[0.4279, 0.7488, 0.7488, 0.5433, 0.5433],
  // [0.4279, 0.7488, 0.8932, 0.9341, 0.9341],
  // [0.964, 0.964, 0.8932, 0.9457, 0.9457],
  // [0.964, 0.964, 0.4822, 0.9874, 0.9874],
  // [0.2132, 0.802, 0.802, 0.9874, 0.9874],
  // [0.8543, 0.8543, 0.802, 0.7816, 0.7816],
  // [0.8543, 0.8543, 0.5178, 0.7816, 0.7816]]]]
  LOG_INFO(dma_1)

  KD::MultidimensionalArrays dma_2 =
      KD::Operator::CreateOperationMaxPool2d(dma_1, {3, 4}, {2, 3}, {0, 1});
  KD::Index t2_size_expect[] = {1, 1, 3, 2};
  KD::BasicData t2_expect[][2] = {
      {0.9640, 0.9457}, {0.9640, 0.9874}, {0.8543, 0.9874}};
  CHECK_EQUAL(dma_2.DimensionsSize(), 4, "check3");
  for (KD::Index i = 0; i < 4; ++i)
    CHECK_EQUAL(dma_2.Size(i), t2_size_expect[i], "check3");
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      KD::BasicData value1 = dma_2[{0, 0, i, j}];
      KD::BasicData value2 = t2_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check3");
    }
  }
  // [[[[0.964, 0.9457],[0.964, 0.9874],[0.8543, 0.9874]]]]
  LOG_INFO(dma_2)

  KD::MultidimensionalArrays mda_3 =
      KD::Operator::CreateOperationImgToCol(dma_0, {4, 4}, {2, 2}, {1, 1});
  KD::Index t3_shape_expect[] = {6, 16};
  CHECK_EQUAL(mda_3.Size(0), t3_shape_expect[0], "check4");
  CHECK_EQUAL(mda_3.Size(1), t3_shape_expect[1], "check4");
  KD::BasicData t3_expect[][16] = {
      {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4279, 0.7488, 0.3639, 0.0000,
       0.2849, 0.6536, 0.8932, 0.0000, 0.9640, 0.4822, 0.1887},
      {0.0000, 0.0000, 0.0000, 0.0000, 0.7488, 0.3639, 0.5433, 0.0000, 0.6536,
       0.8932, 0.9341, 0.0000, 0.4822, 0.1887, 0.9457, 0.0000},
      {0.0000, 0.2849, 0.6536, 0.8932, 0.0000, 0.9640, 0.4822, 0.1887, 0.0000,
       0.2132, 0.0185, 0.0163, 0.0000, 0.2039, 0.8020, 0.3766},
      {0.6536, 0.8932, 0.9341, 0.0000, 0.4822, 0.1887, 0.9457, 0.0000, 0.0185,
       0.0163, 0.9874, 0.0000, 0.8020, 0.3766, 0.6537, 0.0000},
      {0.0000, 0.2132, 0.0185, 0.0163, 0.0000, 0.2039, 0.8020, 0.3766, 0.0000,
       0.8543, 0.3589, 0.5178, 0.0000, 0.0000, 0.0000, 0.0000},
      {0.0185, 0.0163, 0.9874, 0.0000, 0.8020, 0.3766, 0.6537, 0.0000, 0.3589,
       0.5178, 0.7816, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}};
  for (KD::Index i = 0; i < 6; ++i) {
    for (KD::Index j = 0; j < 16; ++j) {
      KD::BasicData value1 = mda_3[{i, j}];
      KD::BasicData value2 = t3_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check4");
    }
  }
  // [[0, 0, 0, 0, 0, 0.4279, 0.7488, 0.3639, 0, 0.2849, 0.6536, 0.8932, 0,
  // 0.964, 0.4822, 0.1887], [0, 0, 0, 0, 0.7488, 0.3639, 0.5433, 0, 0.6536,
  // 0.8932, 0.9341, 0, 0.4822, 0.1887, 0.9457, 0], [0, 0.2849, 0.6536, 0.8932,
  // 0, 0.964, 0.4822, 0.1887, 0, 0.2132, 0.0185, 0.0163, 0, 0.2039, 0.802,
  // 0.3766], [0.6536, 0.8932, 0.9341, 0, 0.4822, 0.1887, 0.9457, 0, 0.0185,
  // 0.0163, 0.9874, 0, 0.802, 0.3766, 0.6537, 0], [0, 0.2132, 0.0185, 0.0163,
  // 0, 0.2039, 0.802, 0.3766, 0, 0.8543, 0.3589, 0.5178, 0, 0, 0, 0], [0.0185,
  // 0.0163, 0.9874, 0, 0.802, 0.3766, 0.6537, 0, 0.3589, 0.5178, 0.7816, 0, 0,
  // 0, 0, 0]]
  LOG_INFO(mda_3)

  KD::BasicData t4_data[2][3][6][4];
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      for (KD::Index k = 0; k < 6; ++k) {
        for (KD::Index l = 0; l < 4; ++l) {
          t4_data[i][j][k][l] = data[k][l];
        }
      }
    }
  }
  KD::MultidimensionalArrays mda_4(reinterpret_cast<KD::BasicData *>(t4_data),
                                   KD::Shape{2, 3, 6, 4});
  // [[[[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]],
  // [[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]],
  // [[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]]],
  // [[[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]],
  // [[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]],
  // [[0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]]]]
  LOG_INFO(mda_4)

  KD::MultidimensionalArrays mda_5 =
      KD::Operator::CreateOperationImgToCol(mda_4, {2, 3}, {1, 2}, {2, 1});
  KD::BasicData t5_expect[18][6] = {
      {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
      {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
      {0.0000, 0.0000, 0.0000, 0.0000, 0.4279, 0.7488},
      {0.0000, 0.0000, 0.0000, 0.7488, 0.3639, 0.5433},
      {0.0000, 0.4279, 0.7488, 0.0000, 0.2849, 0.6536},
      {0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341},
      {0.0000, 0.2849, 0.6536, 0.0000, 0.9640, 0.4822},
      {0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457},
      {0.0000, 0.9640, 0.4822, 0.0000, 0.2132, 0.0185},
      {0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874},
      {0.0000, 0.2132, 0.0185, 0.0000, 0.2039, 0.8020},
      {0.0185, 0.0163, 0.9874, 0.8020, 0.3766, 0.6537},
      {0.0000, 0.2039, 0.8020, 0.0000, 0.8543, 0.3589},
      {0.8020, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816},
      {0.0000, 0.8543, 0.3589, 0.0000, 0.0000, 0.0000},
      {0.3589, 0.5178, 0.7816, 0.0000, 0.0000, 0.0000},
      {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
      {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}};
  CHECK_EQUAL(mda_5.Size(0), 36, "check5");
  CHECK_EQUAL(mda_5.Size(1), 18, "check5");
  for (KD::Index i = 0; i < 36; ++i) {
    for (KD::Index j = 0; j < 18; ++j) {
      KD::BasicData value1 = mda_5[{i, j}];
      KD::BasicData value2 = t5_expect[i / 2][j % 6];
      CHECK_FLOAT_EQUAL(value1, value2, "check5");
    }
  }
  // [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // [0, 0, 0, 0, 0.4279, 0.7488, 0, 0, 0, 0, 0.4279, 0.7488, 0, 0, 0, 0,
  // 0.4279, 0.7488], [0, 0, 0, 0, 0.4279, 0.7488, 0, 0, 0, 0, 0.4279, 0.7488,
  // 0, 0, 0, 0, 0.4279, 0.7488], [0, 0, 0, 0.7488, 0.3639, 0.5433, 0, 0, 0,
  // 0.7488, 0.3639, 0.5433, 0, 0, 0, 0.7488, 0.3639, 0.5433], [0, 0, 0, 0.7488,
  // 0.3639, 0.5433, 0, 0, 0, 0.7488, 0.3639, 0.5433, 0, 0, 0, 0.7488, 0.3639,
  // 0.5433], [0, 0.4279, 0.7488, 0, 0.2849, 0.6536, 0, 0.4279, 0.7488, 0,
  // 0.2849, 0.6536, 0, 0.4279, 0.7488, 0, 0.2849, 0.6536], [0, 0.4279, 0.7488,
  // 0, 0.2849, 0.6536, 0, 0.4279, 0.7488, 0, 0.2849, 0.6536, 0, 0.4279, 0.7488,
  // 0, 0.2849, 0.6536], [0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341,
  // 0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341, 0.7488, 0.3639, 0.5433,
  // 0.6536, 0.8932, 0.9341], [0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341,
  // 0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341, 0.7488, 0.3639, 0.5433,
  // 0.6536, 0.8932, 0.9341], [0, 0.2849, 0.6536, 0, 0.964, 0.4822, 0, 0.2849,
  // 0.6536, 0, 0.964, 0.4822, 0, 0.2849, 0.6536, 0, 0.964, 0.4822], [0, 0.2849,
  // 0.6536, 0, 0.964, 0.4822, 0, 0.2849, 0.6536, 0, 0.964, 0.4822, 0, 0.2849,
  // 0.6536, 0, 0.964, 0.4822], [0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457,
  // 0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457, 0.6536, 0.8932, 0.9341,
  // 0.4822, 0.1887, 0.9457], [0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457,
  // 0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457, 0.6536, 0.8932, 0.9341,
  // 0.4822, 0.1887, 0.9457], [0, 0.964, 0.4822, 0, 0.2132, 0.0185, 0, 0.964,
  // 0.4822, 0, 0.2132, 0.0185, 0, 0.964, 0.4822, 0, 0.2132, 0.0185], [0, 0.964,
  // 0.4822, 0, 0.2132, 0.0185, 0, 0.964, 0.4822, 0, 0.2132, 0.0185, 0, 0.964,
  // 0.4822, 0, 0.2132, 0.0185], [0.4822, 0.1887, 0.9457, 0.0185, 0.0163,
  // 0.9874, 0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874, 0.4822, 0.1887,
  // 0.9457, 0.0185, 0.0163, 0.9874], [0.4822, 0.1887, 0.9457, 0.0185, 0.0163,
  // 0.9874, 0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874, 0.4822, 0.1887,
  // 0.9457, 0.0185, 0.0163, 0.9874], [0, 0.2132, 0.0185, 0, 0.2039, 0.802, 0,
  // 0.2132, 0.0185, 0, 0.2039, 0.802, 0, 0.2132, 0.0185, 0, 0.2039, 0.802], [0,
  // 0.2132, 0.0185, 0, 0.2039, 0.802, 0, 0.2132, 0.0185, 0, 0.2039, 0.802, 0,
  // 0.2132, 0.0185, 0, 0.2039, 0.802], [0.0185, 0.0163, 0.9874, 0.802, 0.3766,
  // 0.6537, 0.0185, 0.0163, 0.9874, 0.802, 0.3766, 0.6537, 0.0185, 0.0163,
  // 0.9874, 0.802, 0.3766, 0.6537], [0.0185, 0.0163, 0.9874, 0.802, 0.3766,
  // 0.6537, 0.0185, 0.0163, 0.9874, 0.802, 0.3766, 0.6537, 0.0185, 0.0163,
  // 0.9874, 0.802, 0.3766, 0.6537], [0, 0.2039, 0.802, 0, 0.8543, 0.3589, 0,
  // 0.2039, 0.802, 0, 0.8543, 0.3589, 0, 0.2039, 0.802, 0, 0.8543, 0.3589], [0,
  // 0.2039, 0.802, 0, 0.8543, 0.3589, 0, 0.2039, 0.802, 0, 0.8543, 0.3589, 0,
  // 0.2039, 0.802, 0, 0.8543, 0.3589], [0.802, 0.3766, 0.6537, 0.3589, 0.5178,
  // 0.7816, 0.802, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816, 0.802, 0.3766,
  // 0.6537, 0.3589, 0.5178, 0.7816], [0.802, 0.3766, 0.6537, 0.3589, 0.5178,
  // 0.7816, 0.802, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816, 0.802, 0.3766,
  // 0.6537, 0.3589, 0.5178, 0.7816], [0, 0.8543, 0.3589, 0, 0, 0, 0, 0.8543,
  // 0.3589, 0, 0, 0, 0, 0.8543, 0.3589, 0, 0, 0], [0, 0.8543, 0.3589, 0, 0, 0,
  // 0, 0.8543, 0.3589, 0, 0, 0, 0, 0.8543, 0.3589, 0, 0, 0], [0.3589, 0.5178,
  // 0.7816, 0, 0, 0, 0.3589, 0.5178, 0.7816, 0, 0, 0, 0.3589, 0.5178, 0.7816,
  // 0, 0, 0], [0.3589, 0.5178, 0.7816, 0, 0, 0, 0.3589, 0.5178, 0.7816, 0, 0,
  // 0, 0.3589, 0.5178, 0.7816, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  LOG_INFO(mda_5)
}

void TestMultidimensionalArraysBackward() {
  KD::BasicData data_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::MultidimensionalArrays mda_1(data_1, KD::Shape{3, 4}, true);
  // [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
  LOG_INFO(mda_1)

  KD::MultidimensionalArrays mda_2 = mda_1.View({2, 2, 3});
  // [[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]]
  LOG_INFO(mda_2)

  KD::MultidimensionalArrays mda_3 = mda_2.Slice(2, 1, 3);
  // [[[2, 3],[5, 6]],[[8, 9],[11, 12]]]
  LOG_INFO(mda_3)

  KD::MultidimensionalArrays mda_4 = mda_3.Slice(1, 1);
  // [[5, 6],[11, 12]]
  LOG_INFO(mda_4)

  mda_4.Backward();
  KD::BasicData t0_grad_expect_1[][4] = {
      {0, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}};
  auto &&grad_1 = mda_1.Grad();
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = grad_1[{i, j}];
      KD::BasicData value2 = t0_grad_expect_1[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
  }
  // [[0, 0, 0, 0],[1, 1, 0, 0],[0, 0, 1, 1]]
  LOG_INFO(grad_1)

  KD::MultidimensionalArrays mda_5 = mda_1.View({3, 2, 2});
  // [[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9, 10],[11, 12]]]
  LOG_INFO(mda_5)

  KD::MultidimensionalArrays mda_6 = mda_5.Transpose(0, 1);
  // [[[1, 2],[5, 6],[9, 10]],[[3, 4],[7, 8],[11, 12]]]
  LOG_INFO(mda_6)

  KD::MultidimensionalArrays mda_7 = mda_6.Slice(0, 0, 1);
  // [[[1, 2],[5, 6],[9, 10]]]
  LOG_INFO(mda_7)

  KD::MultidimensionalArrays mda_8 = mda_7.Permute({1, 2, 0});
  // [[[1],[2]],[[5],[6]],[[9],[10]]]
  LOG_INFO(mda_8)

  mda_8.Backward();
  KD::BasicData t0_grad_expect_2[][4] = {
      {1, 1, 0, 0}, {2, 2, 0, 0}, {1, 1, 1, 1}};
  auto &&grad_2 = mda_1.Grad();
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = grad_2[{i, j}];
      KD::BasicData value2 = t0_grad_expect_2[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }
  // twice grad result.
  // [[1, 1, 0, 0],[2, 2, 0, 0],[1, 1, 1, 1]]
  LOG_INFO(grad_2)
}

void TestBasicOperatorBackward() {
  KD::BasicData data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::MultidimensionalArrays mda_1(data, KD::Shape{3, 4}, true);
  KD::MultidimensionalArrays mda_2(data, KD::Shape{3, 4}, true);
  // [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
  LOG_INFO(mda_1)

  // [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
  LOG_INFO(mda_2)

  KD::MultidimensionalArrays mda_3 = mda_1 + mda_2;
  KD::MultidimensionalArrays mda_4 = mda_1 * (-mda_3);
  KD::MultidimensionalArrays mda_5 = mda_4 - mda_3;
  mda_5.Backward();
  // [[2, 4, 6, 8],[10, 12, 14, 16],[18, 20, 22, 24]]
  LOG_INFO(mda_3)

  // [[-2, -8, -18, -32],[-50, -72, -98, -128],[-162, -200, -242, -288]]
  LOG_INFO(mda_4)

  // [[-4, -12, -24, -40],[-60, -84, -112, -144],[-180, -220, -264, -312]]
  LOG_INFO(mda_5)

  KD::BasicData mda_1_grad_expect[][4] = {
      {-4, -7, -10, -13}, {-16, -19, -22, -25}, {-28, -31, -34, -37}};
  KD::BasicData mda_2_grad_expect[][4] = {
      {-2, -3, -4, -5}, {-6, -7, -8, -9}, {-10, -11, -12, -13}};
  auto &&mda_1_grad = mda_1.Grad();
  auto &&mda_2_grad = mda_2.Grad();
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = mda_1_grad[{i, j}];
      KD::BasicData value2 = mda_2_grad[{i, j}];
      KD::BasicData value3 = mda_1_grad_expect[i][j];
      KD::BasicData value4 = mda_2_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value3, "check1");
      CHECK_FLOAT_EQUAL(value2, value4, "check1");
    }
  }
  // [[-4, -7, -10, -13],[-16, -19, -22, -25],[-28, -31, -34, -37]]
  LOG_INFO(mda_1_grad)

  // [[-2, -3, -4, -5],[-6, -7, -8, -9],[-10, -11, -12, -13]]
  LOG_INFO(mda_2_grad)
}

void TestMatrixOperatorBackward() {
  KD::BasicData data_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::BasicData data_2[] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121};
  KD::MultidimensionalArrays mda_1(data_1, KD::Shape{2, 6}, true);
  KD::MultidimensionalArrays mda_2(data_2, KD::Shape{2, 6}, true);
  // [[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]]
  LOG_INFO(mda_1)

  // [[11, 21, 31, 41, 51, 61],[71, 81, 91, 101, 111, 121]]
  LOG_INFO(mda_2)

  KD::MultidimensionalArrays mda_3 =
      KD::Operator::CreateOperationMatrixMul(mda_1, mda_2.Transpose(0, 1));
  // [[931, 2191],[2227, 5647]]
  LOG_INFO(mda_3)

  mda_3.Backward();
  KD::MultidimensionalArrays mda_4 = mda_1.View({3, 2, 2});
  KD::MultidimensionalArrays mda_5 = mda_2.View({3, 2, 2});
  KD::MultidimensionalArrays mda_6 =
      KD::Operator::CreateOperationBatchMatrixMul(mda_4, mda_5);
  mda_6.Backward();
  // [[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9, 10],[11, 12]]]
  LOG_INFO(mda_4)

  // [[[11, 21],[31, 41]],[[51, 61],[71, 81]],[[91, 101],[111, 121]]]
  LOG_INFO(mda_5)

  // [[[73, 103],[157, 227]],[[681, 791],[925, 1075]],[[1929, 2119],[2333,
  // 2563]]]
  LOG_INFO(mda_6)

  auto &&t1_grad = mda_1.Grad();
  auto &&t2_grad = mda_2.Grad();
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 6; j < 6; ++j) {
      KD::BasicData value1 = t1_grad[{i, j}];
      KD::BasicData value2 = t2_grad[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
  }
  // [[114, 174, 154, 214, 274, 334],[194, 254, 314, 374, 354, 414]]
  LOG_INFO(t1_grad)

  // [[12, 14, 18, 20, 28, 30],[22, 24, 32, 34, 38, 40]]
  LOG_INFO(t2_grad)
}

void TestNumericOperatorBackward() {
  KD::BasicData data_1[] = {0.585639, 0.612628, 0.241485, 0.097616,
                            0.035854, 0.723054, 0.131163, 0.884268,
                            0.193597, 0.694748, 0.650687, 0.738797};
  KD::MultidimensionalArrays dma_1(data_1, KD::Shape{3, 4}, true);
  // [[0.585639, 0.612628, 0.241485, 0.097616],[0.035854, 0.723054, 0.131163,
  // 0.884268],[0.193597, 0.694748, 0.650687, 0.738797]]
  LOG_INFO(dma_1)

  KD::MultidimensionalArrays dma_2 =
      KD::Operator::CreateOperationLogSoftmax(dma_1);
  // [[-1.20896482874321, -1.18197582874321, -1.55311882874321,
  // -1.69698782874321],
  // [-1.86005323576525, -1.17285323576525, -1.76474423576525,
  // -1.01163923576525],
  // [-1.78423866041312, -1.28308766041312, -1.32714866041312,
  // -1.23903866041312]]
  LOG_INFO(dma_2)

  auto labels_ptr =
      KD::Allocator::SharedAllocate<KD::Index>(3 * sizeof(KD::Index));
  auto labels = labels_ptr.get();
  labels[0] = 2, labels[1] = 0, labels[2] = 3;
  KD::MultidimensionalArrays dma_3 =
      KD::Operator::CreateOperationNllLoss(dma_2, labels_ptr);
  // [1.55311882874321, 1.86005323576525, 1.23903866041312]
  LOG_INFO(dma_3)

  dma_3.Backward();
  KD::BasicData dma_1_grad_expect[][4] = {{0.2985, 0.3067, -0.7884, 0.1832},
                                          {-0.8443, 0.3095, 0.1712, 0.3636},
                                          {0.1679, 0.2772, 0.2652, -0.7103}};
  auto &&dma_1_grad = dma_1.Grad();
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = dma_1_grad[{i, j}];
      KD::BasicData value2 = dma_1_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
  }
  // [[0.298506124508601, 0.306672207835006, -0.788412960049184,
  // 0.183234627705578],
  // [-0.84433565676302, 0.309482653349183, 0.171230575541985,
  // 0.363622427871852], [0.167924860188404, 0.277180139678897,
  // 0.26523245192561, -0.710337451792911]]
  LOG_INFO(dma_1_grad)

  KD::BasicData data_2[] = {0.096237, -0.037000, 0.028076, 0.328307,
                            0.122271, -0.017293, 0.150791, 0.421008,
                            0.322066, -0.321352, 0.319534, -0.424081};
  KD::MultidimensionalArrays dma_4(data_2, KD::Shape{2, 2, 3}, true);
  KD::MultidimensionalArrays dma_5 = KD::Operator::CreateOperationMean(
      KD::Operator::CreateOperationSigmoid(
          KD::Operator::CreateOperationRelu(dma_4)),
      1);
  KD::MultidimensionalArrays dma_6 = KD::Operator::CreateOperationMax(dma_5, 1);
  // [[[0.096237, -0.037, 0.028076],[0.328307, 0.122271, -0.017293]],[[0.150791,
  // 0.421008, 0.322066],[-0.321352, 0.319534, -0.424081]]]
  LOG_INFO(dma_4)
  // [[0.552694042632973, 0.515264862011877,
  // 0.503509269484445],[0.518813240662988, 0.591467555207441,
  // 0.539913834749743]]
  LOG_INFO(dma_5)
  // [0.552694042632973, 0.591467555207441]
  LOG_INFO(dma_6)

  dma_6.Backward();
  KD::BasicData dma_2_grad_expect[][2][3] = {
      {{0.1247, 0.0000, 0.0000}, {0.1217, 0.0000, 0.0000}},
      {{0.0000, 0.1196, 0.0000}, {0.0000, 0.1219, 0.0000}}};
  auto &&dma_2_grad = dma_4.Grad();
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 3; ++k) {
        KD::BasicData value1 = dma_2_grad[{i, j, k}];
        KD::BasicData value2 = dma_2_grad_expect[i][j][k];
        CHECK_FLOAT_EQUAL(value1, value2, "check2");
      }
    }
  }
  // [[[0.124711022411849, 0, 0],[0.12169130131953, 0, 0]],[[0,
  // 0.119620621286394, 0],[0, 0.121862834072581, 0]]]
  LOG_INFO(dma_2_grad)
}

void TestImg2colOperatorBackward() {
  KD::BasicData data[6][4] = {
      {0.4279, 0.7488, 0.3639, 0.5433}, {0.2849, 0.6536, 0.8932, 0.9341},
      {0.9640, 0.4822, 0.1887, 0.9457}, {0.2132, 0.0185, 0.0163, 0.9874},
      {0.2039, 0.8020, 0.3766, 0.6537}, {0.8543, 0.3589, 0.5178, 0.7816}};
  KD::MultidimensionalArrays dma_1(reinterpret_cast<KD::BasicData *>(data),
                                   KD::Shape{1, 1, 6, 4}, true);
  // [[[
  // [0.4279, 0.7488, 0.3639, 0.5433],
  // [0.2849, 0.6536, 0.8932, 0.9341],
  // [0.964, 0.4822, 0.1887, 0.9457],
  // [0.2132, 0.0185, 0.0163, 0.9874],
  // [0.2039, 0.802, 0.3766, 0.6537],
  // [0.8543, 0.3589, 0.5178, 0.7816]
  // ]]]
  LOG_INFO(dma_1)

  KD::MultidimensionalArrays dma_2 =
      KD::Operator::CreateOperationImgToCol(dma_1, {5, 3}, {1, 1}, {0, 0});
  dma_2.Backward();
  // [[0.4279, 0.7488, 0.3639, 0.2849, 0.6536, 0.8932, 0.964, 0.4822, 0.1887,
  // 0.2132, 0.0185, 0.0163, 0.2039, 0.802, 0.3766], [0.7488, 0.3639, 0.5433,
  // 0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874,
  // 0.802, 0.3766, 0.6537], [0.2849, 0.6536, 0.8932, 0.964, 0.4822, 0.1887,
  // 0.2132, 0.0185, 0.0163, 0.2039, 0.802, 0.3766, 0.8543, 0.3589, 0.5178],
  // [0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874,
  // 0.802, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816]]
  LOG_INFO(dma_2)

  // [[[[1, 2, 2, 1],[2, 4, 4, 2],[2, 4, 4, 2],[2, 4, 4, 2],[2, 4, 4, 2],[1, 2,
  // 2, 1]]]]
  LOG_INFO(dma_1.Grad())

  KD::MultidimensionalArrays dma_3 =
      KD::Operator::CreateOperationImgToCol(dma_1, {3, 3}, {1, 1}, {1, 1});
  // [[0, 0, 0, 0, 0.4279, 0.7488, 0, 0.2849, 0.6536],
  // [0, 0, 0, 0.4279, 0.7488, 0.3639, 0.2849, 0.6536, 0.8932],
  // [0, 0, 0, 0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341],
  // [0, 0, 0, 0.3639, 0.5433, 0, 0.8932, 0.9341, 0],
  // [0, 0.4279, 0.7488, 0, 0.2849, 0.6536, 0, 0.964, 0.4822],
  // [0.4279, 0.7488, 0.3639, 0.2849, 0.6536, 0.8932, 0.964, 0.4822, 0.1887],
  // [0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457],
  // [0.3639, 0.5433, 0, 0.8932, 0.9341, 0, 0.1887, 0.9457, 0],
  // [0, 0.2849, 0.6536, 0, 0.964, 0.4822, 0, 0.2132, 0.0185],
  // [0.2849, 0.6536, 0.8932, 0.964, 0.4822, 0.1887, 0.2132, 0.0185, 0.0163],
  // [0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874],
  // [0.8932, 0.9341, 0, 0.1887, 0.9457, 0, 0.0163, 0.9874, 0],
  // [0, 0.964, 0.4822, 0, 0.2132, 0.0185, 0, 0.2039, 0.802],
  // [0.964, 0.4822, 0.1887, 0.2132, 0.0185, 0.0163, 0.2039, 0.802, 0.3766],
  // [0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874, 0.802, 0.3766, 0.6537],
  // [0.1887, 0.9457, 0, 0.0163, 0.9874, 0, 0.3766, 0.6537, 0],
  // [0, 0.2132, 0.0185, 0, 0.2039, 0.802, 0, 0.8543, 0.3589],
  // [0.2132, 0.0185, 0.0163, 0.2039, 0.802, 0.3766, 0.8543, 0.3589, 0.5178],
  // [0.0185, 0.0163, 0.9874, 0.802, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816],
  // [0.0163, 0.9874, 0, 0.3766, 0.6537, 0, 0.5178, 0.7816, 0],
  // [0, 0.2039, 0.802, 0, 0.8543, 0.3589, 0, 0, 0],
  // [0.2039, 0.802, 0.3766, 0.8543, 0.3589, 0.5178, 0, 0, 0],
  // [0.802, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816, 0, 0, 0],
  // [0.3766, 0.6537, 0, 0.5178, 0.7816, 0, 0, 0, 0]]
  LOG_INFO(dma_3)

  dma_3.Backward();
  auto &&dma_1_grad = dma_1.Grad();
  KD::BasicData t0_grad_expect[][4] = {{5., 8., 8., 5.},   {8., 13., 13., 8.},
                                       {8., 13., 13., 8.}, {8., 13., 13., 8.},
                                       {8., 13., 13., 8.}, {5., 8., 8., 5.}};
  for (KD::Index i = 0; i < 6; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = dma_1_grad[{0, 0, i, j}];
      KD::BasicData value2 = t0_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
  }
  // [[[[5, 8, 8, 5],[8, 13, 13, 8],[8, 13, 13, 8],[8, 13, 13, 8],[8, 13, 13,
  // 8],[5, 8, 8, 5]]]]
  LOG_INFO(dma_1_grad)
}

void TestBroadcastingOperatorBackward() {
  KD::BasicData data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  KD::MultidimensionalArrays dma_1(data, KD::Shape{1, 3, 4}, true);
  KD::MultidimensionalArrays dma_2(data, KD::Shape{3, 1, 4}, true);

  KD::MultidimensionalArrays dma_3 = dma_1 + dma_2;
  KD::MultidimensionalArrays dma_4 = dma_3 * dma_1;
  KD::MultidimensionalArrays dma_5 = dma_4 - dma_2;
  // [
  // [[2, 4, 6, 8],[6, 8, 10, 12],[10, 12, 14, 16]],
  // [[6, 8, 10, 12],[10, 12, 14, 16],[14, 16, 18, 20]],
  // [[10, 12, 14, 16],[14, 16, 18, 20],[18, 20, 22, 24]]
  // ]
  LOG_INFO(dma_3)

  // [
  // [[2, 8, 18, 32],[30, 48, 70, 96],[90, 120, 154, 192]],
  // [[6, 16, 30, 48],[50, 72, 98, 128],[126, 160, 198, 240]],
  // [[10, 24, 42, 64],[70, 96, 126, 160],[162, 200, 242, 288]]
  // ]
  LOG_INFO(dma_4)

  // [
  // [[1, 6, 15, 28],[29, 46, 67, 92],[89, 118, 151, 188]],
  // [[1, 10, 23, 40],[45, 66, 91, 120],[121, 154, 191, 232]],
  // [[1, 14, 31, 52],[61, 86, 115, 148],[153, 190, 231, 276]]
  // ]
  LOG_INFO(dma_5)

  dma_5.Backward();
  auto &&dma_1_grad = dma_1.Grad();
  auto &&dma_2_grad = dma_2.Grad();
  KD::BasicData dma_1_grad_expect[][4] = {{21.0000, 30.0000, 39.0000, 48.0000},
                                          {45.0000, 54.0000, 63.0000, 72.0000},
                                          {69.0000, 78.0000, 87.0000, 96.0000}};
  KD::BasicData dma_2_grad_expect[] = {12.0000, 15.0000, 18.0000, 21.0000};
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 4; ++j) {
      KD::BasicData value1 = dma_1_grad[{0, i, j}];
      KD::BasicData value2 = dma_1_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
      value1 = dma_2_grad[{i, 0, j}];
      value2 = dma_2_grad_expect[j];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }
  // [[[21, 30, 39, 48],[45, 54, 63, 72],[69, 78, 87, 96]]]
  LOG_INFO(dma_1_grad)

  // [[[12, 15, 18, 21]],[[12, 15, 18, 21]],[[12, 15, 18, 21]]]
  LOG_INFO(dma_2_grad)

  KD::MultidimensionalArrays dma_6(data, KD::Shape{1, 3, 1, 2, 1, 2}, true);
  KD::MultidimensionalArrays dma_7(data, KD::Shape{2, 1, 3, 2, 1, 1}, true);
  // [[[[[[1, 2]],[[3, 4]]]],[[[[5, 6]],[[7, 8]]]],[[[[9, 10]],[[11, 12]]]]]]
  LOG_INFO(dma_6)

  // [[[[[[1]],[[2]]],[[[3]],[[4]]],[[[5]],[[6]]]]],[[[[[7]],[[8]]],[[[9]],[[10]]],[[[11]],[[12]]]]]]
  LOG_INFO(dma_7)

  KD::MultidimensionalArrays dma_8 =
      dma_6 + KD::Operator::CreateOperationSigmoid(
                  dma_6 + KD::Operator::CreateOperationRelu(dma_7));
  // [[[[[[1.88079707797788, 2.95257412682243]],[[3.99330714907572, 4.99752737684337]]],[[[1.98201379003791,
  // 2.99330714907572]],[[3.9990889488056, 4.99966464986953]]],[[[1.99752737684337,
  // 2.9990889488056]],[[3.99987660542401, 4.9999546021313]]]],[[[[5.99752737684337,
  // 6.9990889488056]],[[7.99987660542401, 8.9999546021313]]],[[[5.99966464986953,
  // 6.99987660542401]],[[7.99998329857815, 8.9999938558254]]],[[[5.9999546021313,
  // 6.99998329857815]],[[7.9999977396757, 8.99999916847197]]]],[[[[9.9999546021313,
  // 10.9999832985782]],[[11.9999977396757, 12.999999168472]]],[[[9.9999938558254,
  // 10.9999977396757]],[[11.9999996940978, 12.9999998874648]]],[[[9.99999916847197,
  // 10.9999996940978]],[[11.9999999586006, 12.99999998477]]]]],[[[[[1.99966464986953,
  // 2.99987660542401]],[[3.99998329857815, 4.9999938558254]]],[[[1.9999546021313,
  // 2.99998329857815]],[[3.9999977396757, 4.99999916847197]]],[[[1.9999938558254,
  // 2.9999977396757]],[[3.99999969409777, 4.99999988746484]]]],[[[[5.9999938558254,
  // 6.9999977396757]],[[7.99999969409777, 8.99999988746484]]],[[[5.99999916847197,
  // 6.99999969409777]],[[7.99999995860062, 8.99999998477002]]],[[[5.99999988746484,
  // 6.99999995860062]],[[7.9999999943972, 8.99999999793885]]]],[[[[9.99999988746484,
  // 10.9999999586006]],[[11.9999999943972, 12.9999999979388]]],[[[9.99999998477002,
  // 10.9999999943972]],[[11.9999999992417, 12.9999999997211]]],[[[9.99999999793885,
  // 10.9999999992417]],[[11.9999999998974, 12.9999999999622]]]]]] Shape: (2, 3,
  // 3, 2, 1, 2)
  LOG_INFO(dma_8)

  KD::MultidimensionalArrays dma_9 =
      KD::Operator::CreateOperationMean(dma_6 * dma_7, 0);
  // [[[[[4, 8]],[[15, 20]]],[[[6, 12]],[[21, 28]]],[[[8, 16]],[[27,
  // 36]]]],[[[[20, 24]],[[35, 40]]],[[[30, 36]],[[49, 56]]],[[[40, 48]],[[63,
  // 72]]]],[[[[36, 40]],[[55, 60]]],[[[54, 60]],[[77, 84]]],[[[72, 80]],[[99,
  // 108]]]]] Shape: (3, 3, 2, 1, 2)
  LOG_INFO(dma_9)

  KD::MultidimensionalArrays dma_10 = KD::Operator::CreateOperationMean(
      KD::Operator::CreateOperationMax(dma_9.Squeeze(), 0), 1);
  // [[45.5, 50],[65.5, 72],[85.5, 94]]
  // Shape: (3, 2)
  LOG_INFO(dma_10)

  KD::MultidimensionalArrays dma_11 =
      KD::Operator::CreateOperationLogSoftmax(dma_10);
  // [[-4.51104774484859, -0.0110477448485938],[-6.50150231015975,
  // -0.00150231015975429],[-8.50020344767213, -0.000203447672129439]] Shape:
  // (3, 2)
  LOG_INFO(dma_11)

  KD::MultidimensionalArrays dma_12 = dma_11.View({1, 3, 1, 2, 1, 1});
  // [[[[[[-4.51104774484859]],[[-0.0110477448485938]]]],[[[[-6.50150231015975]],[[-0.00150231015975429]]]],[[[[-8.50020344767213]],[[-0.000203447672129439]]]]]]
  // Shape: (1, 3, 1, 2, 1, 1)
  LOG_INFO(dma_12)

  KD::MultidimensionalArrays dma_13 = dma_8 - dma_6 - dma_12;
  // [[[[[[5.39184482282648, 5.46362187167103]],[[1.00435489392431, 1.00857512169196]]],[[[5.4930615348865,
  // 5.50435489392431]],[[1.01013669365419, 1.01071239471813]]],[[[5.50857512169196,
  // 5.51013669365419]],[[1.01092435027261, 1.01100234697989]]]],[[[[7.49902968700312,
  // 7.50059125896535]],[[1.00137891558377, 1.00145691229105]]],[[[7.50116696002929,
  // 7.50137891558377]],[[1.00148560873791, 1.00149616598515]]],[[[7.50145691229105,
  // 7.50148560873791]],[[1.00150004983546, 1.00150147863173]]]],[[[[9.50015804980343,
  // 9.50018674625028]],[[1.00020118734783, 1.0002026161441]]],[[[9.50019730349753,
  // 9.50020118734783]],[[1.0002031417699, 1.00020333513697]]],[[[9.5002026161441,
  // 9.5002031417699]],[[1.00020340627275, 1.00020343244215]]]]],[[[[[5.51071239471813,
  // 5.51092435027261]],[[1.01103104342675, 1.01104160067399]]],[[[5.51100234697989,
  // 5.51103104342675]],[[1.0110454845243, 1.01104691332057]]],[[[5.51104160067399,
  // 5.5110454845243]],[[1.01104743894637, 1.01104763231343]]]],[[[[7.50149616598515,
  // 7.50150004983546]],[[1.00150200425753, 1.00150219762459]]],[[[7.50150147863173,
  // 7.50150200425753]],[[1.00150226876038, 1.00150229492978]]],[[[7.50150219762459,
  // 7.50150226876038]],[[1.00150230455696, 1.0015023080986]]]],[[[[9.50020333513697,
  // 9.50020340627275]],[[1.00020344206933, 1.00020344561098]]],[[[9.50020343244215,
  // 9.50020344206933]],[[1.00020344691387, 1.00020344739318]]],[[[9.50020344561098,
  // 9.50020344691387]],[[1.00020344756951, 1.00020344763438]]]]]] Shape: (2, 3,
  // 3, 2, 1, 2)
  LOG_INFO(dma_13)

  dma_13.Backward();
  auto &&dma_6_grad = dma_6.Grad();
  auto &&dma_7_grad = dma_7.Grad();
  KD::BasicData dma_6_grad_expect[3][2][2] = {
      {{0.1255, 0.0529}, {0.0077, 0.0029}},
      {{0.0029, 0.0011}, {0.0001, 0.0001}},
      {{-107.3450, 107.3450}, {-125.1927, 125.1927}}};
  KD::BasicData dma_7_grad_expect[2][3][2] = {
      {{3.0877, 2.9434}, {3.0158, 2.9923}, {3.0022, 2.9989}},
      {{2.9345, 2.9341}, {2.9911, 2.9910}, {2.9988, 2.9988}}};
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 2; ++j) {
      for (KD::Index k = 0; k < 2; ++k) {
        KD::BasicData value1 = dma_6_grad[{0, i, 0, j, 0, k}];
        KD::BasicData value2 = dma_6_grad_expect[i][j][k];
        CHECK_FLOAT_EQUAL(value1, value2, "check3");
        value1 = dma_7_grad[{j, 0, i, k, 0, 0}];
        value2 = dma_7_grad_expect[j][i][k];
        CHECK_FLOAT_EQUAL(value1, value2, "check4");
      }
    }
  }
  // [[[[[[0.125509578523502, 0.0528772783936882]],[[0.00770092456490978,
  // 0.00285423096919057]]]],[[[[0.00285423096919057,
  // 0.00105290929349344]],[[0.000142693716168374, 5.25012982066642e-05]]]],[[[[-107.344960140055,
  // 107.345031956478]],[[-125.192711404884, 125.192714980639]]]]]] Shape: (1,
  // 3, 1, 2, 1, 2)
  LOG_INFO(dma_6_grad)

  // [[[[[[3.08768741677299]],[[2.94336477718262]]],[[[3.01577069082022]],[[2.9922616290275]]],[[[3.00221940298326]],[[2.99895136176505]]]]],[[[[[2.93454551962753]],[[2.93410161559743]]],[[[2.99105616167247]],[[2.99099605597267]]],[[[2.99878799934164]],[[2.99877986437328]]]]]]
  // Shape: (2, 1, 3, 2, 1, 1)
  LOG_INFO(dma_7_grad)
}

void TestConv2dModule() {
  KD::BasicData weight_data[] = {
      0.144233,  0.038765,  -0.064723, 0.091522,  -0.221822, 0.243479,
      0.041969,  -0.041030, 0.087458,  0.181160,  -0.175163, 0.031789,
      0.128350,  0.186573,  0.171205,  -0.095062, 0.164999,  -0.001384,
      0.056682,  -0.051798, -0.021868, -0.078280, 0.213687,  0.207394,
      -0.004414, -0.229483, 0.107253,  -0.277729, 0.163448,  0.117666,
      0.083151,  -0.082815, -0.063118, -0.060334, 0.225444,  0.198153};
  KD::Learning::Conv2dWithReLU conv(2, 3, {2, 3}, {2, 1}, {1, 0});
  KD::Learning::ParamsDict params = conv.Parameters();
  KD::MultidimensionalArrays &weight = params["weight"];
  KD::Learning::CpyInitializer initializer(weight, weight_data);
  initializer.Init();

  KD::BasicData img_data[2][2][7][7] = {
      {{{0.906730, 0.916613, 0.722186, 0.386272, 0.100365, 0.618340, 0.609103},
        {0.328955, 0.215732, 0.107681, 0.948013, 0.380048, 0.430663, 0.952055},
        {0.193987, 0.173216, 0.952505, 0.543355, 0.794108, 0.892996, 0.298362},
        {0.364147, 0.519262, 0.255671, 0.286267, 0.373460, 0.638731, 0.768166},
        {0.238655, 0.624588, 0.365848, 0.170788, 0.957593, 0.592034, 0.195187},
        {0.907456, 0.690784, 0.488165, 0.208965, 0.298154, 0.160431, 0.215673},
        {0.082201, 0.544421, 0.673148, 0.754322, 0.053999, 0.834828, 0.282316}},
       {{0.063048, 0.014931, 0.652814, 0.747882, 0.708108, 0.689558, 0.825811},
        {0.642470, 0.979915, 0.715351, 0.195480, 0.097592, 0.995848, 0.853004},
        {0.445599, 0.872031, 0.647845, 0.977787, 0.134027, 0.028618, 0.090089},
        {0.861318, 0.085258, 0.144686, 0.547808, 0.198714, 0.486728, 0.214308},
        {0.133363, 0.289884, 0.341866, 0.149106, 0.463517, 0.422703, 0.378203},
        {0.204468, 0.258783, 0.282937, 0.615461, 0.572807, 0.890848, 0.987701},
        {0.564413, 0.419104, 0.989817, 0.934213, 0.818020, 0.931046,
         0.973152}}},
      {{{0.188894, 0.422389, 0.363913, 0.017957, 0.339406, 0.686006, 0.036082},
        {0.499343, 0.730941, 0.075625, 0.258123, 0.145462, 0.160852, 0.877703},
        {0.273129, 0.795254, 0.169702, 0.263705, 0.574255, 0.985132, 0.046906},
        {0.491382, 0.606278, 0.363986, 0.652343, 0.050938, 0.427025, 0.859248},
        {0.747834, 0.483195, 0.136416, 0.186103, 0.171346, 0.355480, 0.069143},
        {0.427993, 0.046004, 0.433581, 0.905928, 0.488569, 0.371607, 0.220589},
        {0.527133, 0.720200, 0.358453, 0.598218, 0.134067, 0.765214, 0.909041}},
       {{0.518659, 0.629063, 0.760152, 0.316678, 0.783487, 0.723803, 0.640258},
        {0.314762, 0.950146, 0.556054, 0.492749, 0.996876, 0.352247, 0.816742},
        {0.381224, 0.613649, 0.634370, 0.129223, 0.346971, 0.359611, 0.038520},
        {0.098706, 0.768641, 0.262159, 0.443318, 0.791324, 0.880512, 0.713702},
        {0.313552, 0.349498, 0.481868, 0.075583, 0.179302, 0.794912, 0.768620},
        {0.116356, 0.613183, 0.599636, 0.231469, 0.502692, 0.471287, 0.778603},
        {0.297628, 0.129855, 0.114448, 0.860079, 0.003179, 0.437888,
         0.744226}}}};
  KD::MultidimensionalArrays img(reinterpret_cast<KD::BasicData *>(img_data),
                                 KD::Shape{2, 2, 7, 7});
  KD::MultidimensionalArrays out = conv.Forward(img);
  out.Backward();
  LOG_INFO(out)

  KD::BasicData out_expect[2][3][4][5] = {
      {{{0.085057, 0.000000, 0.014622, 0.197015, 0.054076},
        {0.257979, 0.015247, 0.168568, 0.460462, 0.017097},
        {0.058030, 0.126796, 0.304079, 0.000000, 0.061810},
        {0.259736, 0.174524, 0.045041, 0.427429, 0.013675}},
       {{0.197690, 0.324926, 0.250511, 0.214764, 0.354828},
        {0.365562, 0.637203, 0.468079, 0.286736, 0.313169},
        {0.430900, 0.230755, 0.218951, 0.541876, 0.418846},
        {0.652668, 0.632795, 0.476516, 0.326459, 0.538814}},
       {{0.111889, 0.203388, 0.143905, 0.233037, 0.421233},
        {0.272305, 0.544619, 0.000000, 0.000000, 0.000000},
        {0.165981, 0.000000, 0.071176, 0.337920, 0.000000},
        {0.269469, 0.297138, 0.173952, 0.105476, 0.404562}}},
      {{{0.020134, 0.000000, 0.219107, 0.036524, 0.000000},
        {0.000000, 0.255312, 0.301976, 0.153720, 0.000000},
        {0.071727, 0.160195, 0.229189, 0.204703, 0.000000},
        {0.078193, 0.149058, 0.000000, 0.536360, 0.151846}},
       {{0.302706, 0.198736, 0.138555, 0.346089, 0.306997},
        {0.507746, 0.232753, 0.143462, 0.263327, 0.384909},
        {0.357071, 0.345307, 0.184858, 0.338571, 0.574458},
        {0.195879, 0.423052, 0.559256, 0.236188, 0.516787}},
       {{0.320550, 0.140453, 0.122583, 0.432157, 0.264884},
        {0.065698, 0.000000, 0.021695, 0.197138, 0.133571},
        {0.000000, 0.010505, 0.000000, 0.158508, 0.281115},
        {0.002592, 0.101532, 0.043178, 0.000000, 0.330650}}}};
  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      for (KD::Index k = 0; k < 4; ++k) {
        for (KD::Index l = 0; l < 5; ++l) {
          KD::BasicData value1 = out[{i, j, k, l}];
          KD::BasicData value2 = out_expect[i][j][k][l];
          CHECK_FLOAT_EQUAL(value1, value2, "check1");
        }
      }
    }
  }

  auto &&weight_grad = weight.Grad();
  LOG_INFO(weight_grad)

  KD::BasicData weight_grad_expect[3][12] = {
      {11.133665, 9.121082, 9.863847, 14.400089, 14.734153, 16.512037,
       10.890715, 13.367422, 13.612727, 15.687311, 14.727955, 17.525148},
      {12.549257, 11.719290, 12.803723, 17.626467, 20.197935, 17.964199,
       14.141121, 15.980512, 16.688646, 18.285843, 19.956493, 21.097359},
      {7.728555, 7.243347, 8.941869, 11.476597, 16.036528, 14.276077, 11.306244,
       11.789472, 11.962565, 13.503635, 16.988863, 19.089035}};
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 12; ++j) {
      KD::BasicData value1 = weight_grad[{i, j}];
      KD::BasicData value2 = weight_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }
}

void TestLinearModule() {
  KD::BasicData weight_data[6][5] = {
      0.071760,  0.263576,  -0.378940, -0.424306, 0.424915,  0.406897,
      0.142503,  0.361772,  -0.061179, 0.132496,  0.226302,  0.022161,
      -0.021480, -0.283614, -0.442592, 0.032238,  -0.245419, -0.083803,
      -0.155786, 0.081459,  -0.104956, 0.009876,  0.175388,  0.024486,
      -0.188793, 0.262046,  -0.425379, -0.263474, 0.102063,  -0.067243};
  KD::BasicData bias_data[] = {0.053275, 0.057604,  -0.233080,
                               0.186017, -0.003390, 0.101612};
  KD::Learning::LinearWithReLU linear(5, 6);
  KD::Learning::ParamsDict params = linear.Parameters();
  KD::MultidimensionalArrays &weight = params["weight"];
  KD::MultidimensionalArrays &bias = params["bias"];
  KD::Learning::CpyInitializer weight_initializer(
      weight, reinterpret_cast<KD::BasicData *>(weight_data));
  KD::Learning::CpyInitializer bias_initializer(
      bias, reinterpret_cast<KD::BasicData *>(bias_data));
  weight_initializer.Init();
  bias_initializer.Init();

  KD::BasicData input_data[3][5] = {
      {0.524644, 0.069943, 0.090128, 0.390283, 0.264224},
      {0.360333, 0.167909, 0.272388, 0.330552, 0.947953},
      {0.735467, 0.036351, 0.184947, 0.862948, 0.818394}};
  KD::MultidimensionalArrays input(
      reinterpret_cast<KD::BasicData *>(input_data), KD::Shape{3, 5});

  KD::MultidimensionalArrays linear_out = linear.Forward(input);
  KD::MultidimensionalArrays out =
      KD::Operator::CreateOperationLogSoftmax(linear_out);
  out.Backward();
  LOG_INFO(out)

  KD::BasicData out_expect[3][6] = {
      {-1.892938, -1.590033, -1.914817, -1.775882, -1.914817, -1.707157},
      {-1.672153, -1.522798, -1.954868, -1.795545, -1.954868, -1.932030},
      {-1.929682, -1.472234, -1.956825, -1.839288, -1.956825, -1.693635}};
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 6; ++j) {
      KD::BasicData value1 = out[{i, j}];
      KD::BasicData value2 = out_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
  }
  KD::BasicData weight_grad_expect[6][5] = {
      {0.099457, -0.009920, -0.002108, 0.106735, 0.010422},
      {-0.505351, -0.081136, -0.173833, -0.514122, -0.659705},
      {0.000000, 0.000000, 0.000000, 0.000000, 0.000000},
      {0.027103, 0.001202, 0.008172, 0.035058, 0.037341},
      {0.000000, 0.000000, 0.000000, 0.000000, 0.000000},
      {-0.074985, 0.012053, 0.008624, -0.080164, 0.016362}};
  auto &&weight_grad = weight.Grad();
  LOG_INFO(weight_grad)

  for (KD::Index i = 0; i < 6; ++i) {
    for (KD::Index j = 0; j < 5; ++j) {
      KD::BasicData value1 = weight_grad[{i, j}];
      KD::BasicData value2 = weight_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }
  KD::BasicData bias_grad_expect[6] = {0.098009, -0.908593, 0.000000,
                                       0.034192, 0.000000,  -0.060508};
  auto &&bias_grad = bias.Grad();
  LOG_INFO(bias_grad)

  for (KD::Index i = 0; i < 6; ++i) {
    KD::BasicData value1 = bias_grad[{0, i}];
    KD::BasicData value2 = bias_grad_expect[i];
    CHECK_FLOAT_EQUAL(value1, value2, "check3");
  }
}

void TestMaxPool2dModule() {
  KD::BasicData data1[2][6] = {
      {0.138318, 0.883046, 0.093294, 0.514822, 0.359068, 0.650812},
      {0.576113, 0.390465, 0.855900, 0.452224, 0.551624, 0.140468}};
  KD::BasicData data2[2][7] = {
      {0.574436, 0.286016, 0.286861, 0.392806, 0.088330, 0.456134, 0.482773},
      {0.387206, 0.814651, 0.888812, 0.004778, 0.971438, 0.481807, 0.931557}};
  KD::MultidimensionalArrays dma_1(reinterpret_cast<KD::BasicData *>(data1),
                                   KD::Shape{1, 2, 1, 6}, true);
  KD::MultidimensionalArrays dma_2(reinterpret_cast<KD::BasicData *>(data2),
                                   KD::Shape{2, 1, 7, 1}, true);
  KD::MultidimensionalArrays img = dma_1 + dma_2;

  KD::Learning::MaxPool2d max_pool({3, 2}, {1, 2}, {1, 0});
  KD::MultidimensionalArrays max_pool_output = max_pool.Forward(img);
  KD::MultidimensionalArrays reduced = KD::Operator::CreateOperationMean(
      KD::Operator::CreateOperationMean(max_pool_output, 0), 2);
  KD::MultidimensionalArrays out =
      KD::Operator::CreateOperationMatrixMul(reduced, dma_2.View({7, 2}));
  out.Backward();
  LOG_INFO(out)

  KD::BasicData output_expect[2][2] = {{3.787218, 5.987955},
                                       {3.727950, 5.894424}};
  for (KD::Index i = 0; i < 2; ++i)
    for (KD::Index j = 0; j < 2; ++j) {
      KD::BasicData value1 = out[{i, j}];
      KD::BasicData value2 = output_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }

  KD::BasicData t0_grad_expect[2][6] = {
      {0.000000, 2.349201, 0.000000, 2.349201, 0.000000, 2.349201},
      {2.349201, 0.000000, 2.349201, 0.000000, 2.349201, 0.000000}};
  auto &&t0_grad = dma_1.Grad();
  LOG_INFO(t0_grad)

  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 6; j < 6; ++j) {
      KD::BasicData value1 = t0_grad[{0, i, 0, j}];
      KD::BasicData value2 = t0_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }

  KD::BasicData t1_grad_expect[2][7] = {
      {4.273312, 2.733193, 2.807353, 4.221796, 2.625723, 4.329186, 5.097929},
      {2.708350, 3.632129, 3.995808, 2.798316, 6.347974, 2.758435, 4.171799}};
  auto &&dma_2_grad = dma_2.Grad();
  LOG_INFO(dma_2_grad)

  for (KD::Index i = 0; i < 2; ++i) {
    for (KD::Index j = 0; j < 7; ++j) {
      KD::BasicData value1 = dma_2_grad[{i, 0, j, 0}];
      KD::BasicData value2 = t1_grad_expect[i][j];
      CHECK_FLOAT_EQUAL(value1, value2, "check3");
    }
  }
}

void TestCrossEntropyModule() {
  KD::BasicData weight_data[3][5] = {
      {0.332016, 0.383861, -0.039896, -0.286464, 0.069793},
      {-0.341369, 0.439378, -0.156823, -0.208273, -0.401472},
      {0.201601, 0.154146, -0.086722, -0.359864, 0.297248}};
  KD::BasicData bias_data[3] = {-0.240007, 0.322247, -0.051916};
  KD::Learning::Linear linear(5, 3);
  KD::Learning::ParamsDict params = linear.Parameters();
  KD::MultidimensionalArrays &weight = params["weight"];
  KD::MultidimensionalArrays &bias = params["bias"];
  KD::Learning::CpyInitializer weight_initializer(
      weight, reinterpret_cast<KD::BasicData *>(weight_data));
  KD::Learning::CpyInitializer bias_initializer(
      bias, reinterpret_cast<KD::BasicData *>(bias_data));
  weight_initializer.Init();
  bias_initializer.Init();

  KD::BasicData input_data[3][5] = {
      {0.521807, 0.487334, 0.844843, 0.366452, 0.744550},
      {0.861821, 0.102663, 0.949307, 0.086492, 0.588144},
      {0.788253, 0.402394, 0.554831, 0.984794, 0.170077}};
  KD::Index labels[] = {2, 1, 0};
  KD::MultidimensionalArrays input(
      reinterpret_cast<KD::BasicData *>(input_data), KD::Shape{3, 5});

  KD::Learning::CrossEntropy criterion;
  KD::MultidimensionalArrays out = linear.Forward(input);
  KD::MultidimensionalArrays loss = criterion.Forward(out, labels);
  loss.Backward();

  KD::BasicData loss_expect = 1.157702;
  CHECK_FLOAT_EQUAL(loss_expect, loss.Item(), "check1");

  KD::BasicData weight_grad_expect[3][5] = {
      {-0.011948, -0.021014, 0.086070, -0.164264, 0.116384},
      {-0.080774, 0.065085, -0.098823, 0.123327, -0.059957},
      {0.092722, -0.044071, 0.012753, 0.040937, -0.056427}};
  auto &&weight_grad = weight.Grad();
  for (KD::Index i = 0; i < 3; ++i) {
    for (KD::Index j = 0; j < 5; ++j) {
      KD::BasicData value1 = weight_grad_expect[i][j];
      KD::BasicData value2 = weight_grad[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check2");
    }
  }

  KD::BasicData bias_grad_expect[3] = {0.012001, -0.047002, 0.035001};
  auto &&bias_grad = bias.Grad();
  for (KD::Index i = 0; i < 3; ++i) {
    KD::BasicData value1 = bias_grad_expect[i];
    KD::BasicData value2 = bias_grad[{0, i}];
    CHECK_FLOAT_EQUAL(value1, value2, "check3");
  }
  LOG_INFO(weight_grad)
  LOG_INFO(bias_grad)
}

void TestOptimizer() {
  KD::BasicData weight_data[4][3] = {{0.5437, -0.4394, -0.0307},
                                     {-0.3073, 0.4709, 0.1285},
                                     {-0.0405, 0.5013, -0.3253},
                                     {0.4171, -0.2727, -0.3348}};
  KD::BasicData bias_data[4] = {0.1618, -0.4150, 0.1099, 0.2695};
  KD::Learning::Linear linear(3, 4);
  KD::Learning::ParamsDict params = linear.Parameters();
  KD::MultidimensionalArrays &weight = params["weight"];
  KD::MultidimensionalArrays &bias = params["bias"];
  KD::Learning::CpyInitializer weight_initializer(
      weight, reinterpret_cast<KD::BasicData *>(weight_data));
  KD::Learning::CpyInitializer bias_initializer(
      bias, reinterpret_cast<KD::BasicData *>(bias_data));
  weight_initializer.Init();
  bias_initializer.Init();

  KD::Learning::StochasticGradientDescentWithMomentum optimizer(
      linear.Parameters(), 0.01, 0.9);

  KD::BasicData input_data[2][3] = {{0.4746, 0.5383, 0.2668},
                                    {0.0405, 0.8955, 0.7365}};
  KD::MultidimensionalArrays input(
      reinterpret_cast<KD::BasicData *>(input_data), KD::Shape{2, 3});

  KD::MultidimensionalArrays out1 = linear.Forward(input);
  out1.Backward();
  optimizer.Step();
  optimizer.ZeroGrad();
  LOG_INFO(weight)
  LOG_INFO(bias)
  KD::BasicData weight_expect1[4][3] = {{0.5385, -0.4537, -0.0407},
                                        {-0.3125, 0.4566, 0.1185},
                                        {-0.0457, 0.4870, -0.3354},
                                        {0.4119, -0.2871, -0.3448}};
  KD::BasicData bias_expect1[4] = {0.1418, -0.4350, 0.0899, 0.2495};
  for (KD::Index i = 0; i < 4; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = weight_expect1[i][j];
      KD::BasicData value2 = weight[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
    KD::BasicData value1 = bias_expect1[i];
    KD::BasicData value2 = bias[{0, i}];
    CHECK_FLOAT_EQUAL(value1, value2, "check1");
  }

  KD::MultidimensionalArrays out2 = linear.Forward(input);
  out2.Backward();
  optimizer.Step();
  optimizer.ZeroGrad();
  LOG_INFO(weight)
  LOG_INFO(bias)
  KD::BasicData weight_expect2[4][3] = {{0.5287, -0.4809, -0.0598},
                                        {-0.3223, 0.4293, 0.0994},
                                        {-0.0555, 0.4597, -0.3544},
                                        {0.4022, -0.3143, -0.3639}};
  KD::BasicData bias_expect2[4] = {0.1038, -0.4730, 0.0519, 0.2115};
  for (KD::Index i = 0; i < 4; ++i) {
    for (KD::Index j = 0; j < 3; ++j) {
      KD::BasicData value1 = weight_expect2[i][j];
      KD::BasicData value2 = weight[{i, j}];
      CHECK_FLOAT_EQUAL(value1, value2, "check1");
    }
    KD::BasicData value1 = bias_expect2[i];
    KD::BasicData value2 = bias[{0, i}];
    CHECK_FLOAT_EQUAL(value1, value2, "check1");
  }
}