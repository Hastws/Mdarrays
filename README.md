![logo](./docs/images/icon.png)

# 多维数组 & 自动求导（C++11）
> 基础机器学习组件 | 多维数组计算 | 自动求导 | 线性/卷积/池化模块 | 简易优化器

本项目实现了一个轻量级的多维数组（`Mdarray`）与表达式系统，支持**惰性计算 + 自动求导**，并在其上构建了常见的机器学习原语与模块（`Linear`、`Conv2d`、`MaxPool2d`、`CrossEntropy`、`SGD+Momentum` 等）。仓库内附带完整自测：编译后直接运行主程序即可验证功能与梯度正确性。

---

## ✨ 功能一览
- **多维数组运算**：逐元素加减乘、广播、矩阵乘/转置、批矩阵乘/转置
- **形状/视图**：`View`、`Slice`、`Unsqueeze/Squeeze`、`Permute`、`Transpose`
- **数值算子**：`Relu`、`Sigmoid`、`Mean`、`Max`、`Argmax`、`LogSoftmax`、`NllLoss`
- **卷积与池化**：`Img2Col`、`Conv2d(+ReLU)`、`MaxPool2d`（含反向）
- **模块与优化器**：`Linear(+ReLU)`、`Conv2d(+ReLU)`、`CrossEntropy`、`SGD+Momentum`
- **自动求导**：对 `requires_grad=true` 的张量构图，`Backward()` 触发回传；`.Grad()` 读取梯度
- **日志与断言**：`LOG_MDA_INFO` 打印张量/形状，`CHECK_*` 校验数值（见 `utils/exception.h`）

---

## 🔧 环境要求
- C++11
- CMake ≥ 3.5
- 第三方依赖：**无**

---

## ⚡ 快速开始（60 秒）
> ```bash
> mkdir build
> cd build
> cmake ..
> make -j
> ```

---

## 🧪 自测说明
`main()` 中调用 `TestLazyEvaluation()`，其内部顺序执行以下用例（全部通过即认为组件可用）：
- `TestMdarray / TestBasicOperator / TestMatrixOperator / TestNumericOperator / TestConvOperator`
- `TestMdarrayBackward / TestBasicOperatorBackward / TestMatrixOperatorBackward / TestNumericOperatorBackward / TestImg2colOperatorBackward / TestBroadcastingOperatorBackward`
- `TestConv2dModule / TestLinearModule / TestMaxPool2dModule / TestCrossEntropyModule / TestOptimizer`

每个用例在核心步骤会 `LOG_MDA_INFO(...)` 打印张量/形状，并用 `CHECK_*` 断言期望值（断言失败会抛异常并打印上下文）。

---

## 🧩 API 速览

### 构造与索引
```cpp
Autoalg::BasicData data[] = {1,2,3,4,5,6};
Autoalg::Mdarray x(data, Autoalg::Shape{2,3}, /*requires_grad=*/true);
auto v = x[{1,2}];
```

### 形状操作
```cpp
auto xT = x.Transpose(0,1);
auto y  = x.View({3,2}).Unsqueeze(0).Squeeze();
auto z  = x.Permute({1,0});
auto s  = x.Slice(1, 0, 2);  // 对第1维切片 [0,2)
```

### 数值与矩阵算子
```cpp
auto logp = Autoalg::Operator::CreateOperationLogSoftmax(x);
auto m0   = Autoalg::Operator::CreateOperationMean(x, 0);
auto mx   = Autoalg::Operator::CreateOperationMax(x, 1);
auto idx  = Autoalg::Operator::CreateOperationArgmax(x, 1);

auto mm   = Autoalg::Operator::CreateOperationMatrixMul(x, xT);
auto XT   = Autoalg::Operator::CreateOperationMatrixTranspose(x);
```

### 训练模块与优化
```cpp
Autoalg::Learning::LinearWithReLU fc(5, 6);
Autoalg::Learning::CrossEntropy criterion;
Autoalg::Learning::StochasticGradientDescentWithMomentum optim(
  fc.Parameters(), /*lr=*/0.01, /*momentum=*/0.9
);

Autoalg::Mdarray logits = fc.Forward(input);         // input: (N,5)
Autoalg::Mdarray loss   = criterion.Forward(logits, labels); // labels: N
loss.Backward();
optim.Step();
optim.ZeroGrad();
```

### 卷积与池化
```cpp
Autoalg::Learning::Conv2dWithReLU conv(2, 3, /*kernel=*/{2,3}, /*stride=*/{2,1}, /*pad=*/{1,0});
Autoalg::Learning::MaxPool2d pool(/*kernel=*/{3,2}, /*stride=*/{1,2}, /*pad=*/{1,0});
Autoalg::Mdarray y = conv.Forward(img);  // img: (N,C,H,W)
Autoalg::Mdarray z = pool.Forward(y);
z.Backward();
```

> **自动求导要点**
> - 构造张量时传 `requires_grad=true` 以参与反向。
> - 任一标量/张量调用 `Backward()` 后，可通过 `.Grad()` 获取梯度。
> - 断言/异常在 `utils/exception.h` 中定义。

---

## 🚀 性能建议
- 使用 `Release/RelWithDebInfo`（`-O3` 或 `/O2`）
- 纯 CPU 推理

---

## 🐞 FAQ
**Q：断言失败怎么办？**  
A：查看相应 `CHECK_*` 的提示与 `LOG_MDA_INFO` 输出；重点检查输入形状与广播维度。

**Q：只想跑某个 Test？**  
A：在 `TestLazyEvaluation()` 中注释/启用对应 `Test*` 调用。

**Q：如何自定义初始权重？**  
A：使用 `Autoalg::Learning::CpyInitializer` 对 `ParamsDict` 中的 `weight/bias` 赋值（见 `TestLinearModule()` / `TestConv2dModule()`）。

---

## 📜 许可证
GNU GENERAL PUBLIC LICENSE VERSION 3

---
