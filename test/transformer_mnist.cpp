// transformer_mnist.cc
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "data/data.h"
#include "exp/function.h"  // 包含 CreateOperationSoftmax / Mean / Argmax / BatchMatMul 等
#include "learning/module.h"
#include "learning/optimizer.h"
#include "mdarray/mdarray.h"
#include "mdarray/shape.h"
#include "utils/exception.h"

// ====== Multi-Head Self-Attention ======
class MultiHeadSelfAttention : public Autoalg::Learning::Module {
 public:
  MultiHeadSelfAttention(Autoalg::Index d_model, Autoalg::Index n_heads)
      : d_model_(d_model),
        n_heads_(n_heads),
        d_head_(d_model / n_heads),
        q_proj_(d_model, d_model),
        k_proj_(d_model, d_model),
        v_proj_(d_model, d_model),
        o_proj_(d_model, d_model) {
    CHECK_TRUE(d_model_ % n_heads_ == 0,
               "d_model must be divisible by n_heads");
  }

  // 可选：把任意张量“实体化”为连续内存，避免对非连续 view 再次 view 出问题
  inline Autoalg::Mdarray MakeContiguous(const Autoalg::Mdarray& x) {
    Autoalg::Mdarray y(x.Size(), true);
    y = x; // 触发 Assign，把数据拷到连续存储
    return y;
  }

  // 沿最后一维做 softmax：把 (..., C) 视作 (N, C)
  inline Autoalg::Mdarray SoftmaxLastDim(const Autoalg::Mdarray& x, bool force_contiguous = true) {
    Autoalg::Mdarray src = force_contiguous ? MakeContiguous(x) : x;

    Autoalg::Index dims = src.DimensionsSize();
    CHECK_TRUE(dims >= 1, "SoftmaxLastDim expects rank >= 1");

    Autoalg::Index C = src.Size(dims - 1);
    Autoalg::Index N = 1;
    for (Autoalg::Index i = 0; i < dims - 1; ++i) N *= src.Size(i);

    Autoalg::Mdarray x2d = src.View({N, C});                         // (N, C)
    Autoalg::Mdarray y2d(x2d.Size(), true);
    y2d = Autoalg::Operator::CreateOperationSoftmax(x2d);        // 复用你的 2D softmax
    return y2d.View(src.Size());                            // 还原形状 (..., C)
  }

  Autoalg::Mdarray Forward(const Autoalg::Mdarray& x_bsd) override {
    // x_bsd: (B, S, D)
    const Autoalg::Index B = x_bsd.Size(0);
    const Autoalg::Index S = x_bsd.Size(1);
    const Autoalg::Index D = x_bsd.Size(2);

    // Linear projections (flatten batch*seq for Linear)
    Autoalg::Mdarray x_2d = x_bsd.View({B * S, D});
    Autoalg::Mdarray q = q_proj_.Forward(x_2d).View({B, S, D});
    Autoalg::Mdarray k = k_proj_.Forward(x_2d).View({B, S, D});
    Autoalg::Mdarray v = v_proj_.Forward(x_2d).View({B, S, D});

    // Reshape to heads: (B, H, S, Dh)
    Autoalg::Mdarray q_bhsd =
        q.View({B, S, n_heads_, d_head_}).Permute({0, 2, 1, 3});
    Autoalg::Mdarray k_bhsd =
        k.View({B, S, n_heads_, d_head_}).Permute({0, 2, 1, 3});
    Autoalg::Mdarray v_bhsd =
        v.View({B, S, n_heads_, d_head_}).Permute({0, 2, 1, 3});

    // Merge (B,H) -> BH for batch matmul
    Autoalg::Mdarray q_bhsd_2 =
        q_bhsd.Contiguous().View({B * n_heads_, S, d_head_});  // (BH, S, Dh)
    Autoalg::Mdarray k_bhsd_2 =
        k_bhsd.Contiguous().View({B * n_heads_, S, d_head_});  // (BH, S, Dh)
    Autoalg::Mdarray v_bhsd_2 =
        v_bhsd.Contiguous().View({B * n_heads_, S, d_head_});  // (BH, S, Dh)

    // scores = (Q @ K^T) / sqrt(Dh)  -> (BH, S, S)
    Autoalg::Mdarray kt = Autoalg::Operator::CreateOperationBatchMatrixTranspose(
        k_bhsd_2);  // (BH, Dh, S)
    Autoalg::Mdarray scores = Autoalg::Operator::CreateOperationBatchMatrixMul(
        q_bhsd_2, kt);  // (BH, S, S)

    // scale by 1/sqrt(Dh)
    const Autoalg::BasicData scale = 1.0 / std::sqrt(static_cast<double>(d_head_));
    Autoalg::Mdarray scale_const = Autoalg::Operator::CreateOperationConstant(
        scale, {scores.Size(0), scores.Size(1), scores.Size(2)});
    scores = scores * scale_const;

    // attn = softmax(scores, dim=-1)
    Autoalg::Mdarray attn = SoftmaxLastDim(scores);
    std::cout << attn << std::endl;
    // Autoalg::Mdarray attn =
    //     Autoalg::Operator::CreateOperationSoftmax(scores2);  // (BH, S, S)

    // ctx = attn @ V  -> (BH, S, Dh)
    Autoalg::Mdarray ctx =
        Autoalg::Operator::CreateOperationBatchMatrixMul(attn, v_bhsd_2);

    // reshape back to (B, S, D)
    Autoalg::Mdarray ctx_bhsd = ctx.View({B, n_heads_, S, d_head_})
                               .Permute({0, 2, 1, 3}).Contiguous();  // (B, S, H, Dh)
    Autoalg::Mdarray ctx_bsd = ctx_bhsd.View({B * S, d_model_});
    Autoalg::Mdarray out_bsd = o_proj_.Forward(ctx_bsd).View({B, S, d_model_});
    return out_bsd;
  }

  Autoalg::Learning::ParamsDict Parameters() override {
    return {{"q_proj", q_proj_.Parameters()},
            {"k_proj", k_proj_.Parameters()},
            {"v_proj", v_proj_.Parameters()},
            {"o_proj", o_proj_.Parameters()}};
  }

 private:
  Autoalg::Index d_model_, n_heads_, d_head_;
  Autoalg::Learning::Linear q_proj_, k_proj_, v_proj_, o_proj_;
};

// ====== Position-wise FeedForward ======
class FeedForward : public Autoalg::Learning::Module {
 public:
  FeedForward(Autoalg::Index d_model, Autoalg::Index d_hidden)
      : fc1_(d_model, d_hidden), fc2_(d_hidden, d_model) {}

  Autoalg::Mdarray Forward(const Autoalg::Mdarray& x_bsd) override {
    // flatten to (B*S, D) for LinearWithReLU/Linear
    Autoalg::Index B = x_bsd.Size(0), S = x_bsd.Size(1), D = x_bsd.Size(2);
    Autoalg::Mdarray x2d = x_bsd.View({B * S, D});
    Autoalg::Mdarray h = fc1_.Forward(x2d);  // has ReLU inside
    Autoalg::Mdarray y2d = fc2_.Forward(h);
    return y2d.View({B, S, D});
  }

  Autoalg::Learning::ParamsDict Parameters() override {
    return {{"fc1", fc1_.Parameters()}, {"fc2", fc2_.Parameters()}};
  }

 private:
  Autoalg::Learning::LinearWithReLU fc1_;
  Autoalg::Learning::Linear fc2_;
};

// ====== Transformer Encoder Block (简化版，无 LayerNorm) ======
class TransformerEncoderBlock : public Autoalg::Learning::Module {
 public:
  TransformerEncoderBlock(Autoalg::Index d_model, Autoalg::Index n_heads, Autoalg::Index d_ff)
      : attn_(d_model, n_heads), ffn_(d_model, d_ff) {}

  Autoalg::Mdarray Forward(const Autoalg::Mdarray& x_bsd) override {
    Autoalg::Mdarray a = attn_.Forward(x_bsd);
    Autoalg::Mdarray x1 = x_bsd + a;  // residual 1
    Autoalg::Mdarray f = ffn_.Forward(x1);
    Autoalg::Mdarray x2 = x1 + f;  // residual 2
    return x2;
  }

  Autoalg::Learning::ParamsDict Parameters() override {
    return {{"attn", attn_.Parameters()}, {"ffn", ffn_.Parameters()}};
  }

 private:
  MultiHeadSelfAttention attn_;
  FeedForward ffn_;
};

// ====== Sinusoidal Positional Encoding（非参数，保存在 buffer 内） ======
static inline void BuildSinusoidalPE(std::vector<Autoalg::BasicData>& buf,
                                     Autoalg::Index seq_len, Autoalg::Index d_model) {
  buf.resize(seq_len * d_model);
  for (Autoalg::Index pos = 0; pos < seq_len; ++pos) {
    for (Autoalg::Index i = 0; i < d_model; ++i) {
      double div =
          std::pow(10000.0, (i / 2) * 2.0 / static_cast<double>(d_model));
      double val = pos / div;
      double v = (i % 2 == 0) ? std::sin(val) : std::cos(val);
      buf[pos * d_model + i] = static_cast<Autoalg::BasicData>(v);
    }
  }
}

// ====== Transformer for MNIST rows-as-tokens ======
class TransformerMNIST : public Autoalg::Learning::Module {
 public:
  TransformerMNIST(Autoalg::Index in_per_token,  // 28
                   Autoalg::Index seq_len,       // 28 (image rows)
                   Autoalg::Index d_model,       // 64
                   Autoalg::Index n_heads,       // 4
                   Autoalg::Index d_ff,          // 128
                   Autoalg::Index n_layers,      // 2
                   Autoalg::Index n_classes)     // 10
      : in_per_token_(in_per_token),
        seq_len_(seq_len),
        d_model_(d_model),
        n_heads_(n_heads),
        d_ff_(d_ff),
        n_layers_(n_layers),
        tok_proj_(in_per_token, d_model),
        head_(d_model, n_classes),
        pe_buf_(),
        pe_(Autoalg::Shape({1, seq_len, d_model})) {
    // build PE buffer and Mdarray view
    BuildSinusoidalPE(pe_buf_, seq_len_, d_model_);
    pe_ = Autoalg::Mdarray(pe_buf_.data(), Autoalg::Shape({1, seq_len_, d_model_}));

    // create blocks (here fixed to 2 for simplicity; adjust as needed)
    blocks_.emplace_back(
        new TransformerEncoderBlock(d_model_, n_heads_, d_ff_));
    blocks_.emplace_back(
        new TransformerEncoderBlock(d_model_, n_heads_, d_ff_));
  }

  Autoalg::Mdarray Forward(const Autoalg::Mdarray& input) override {
    // input: (B, 784). 我们把每一行(28)当作一个token，序列长度 S=28
    const Autoalg::Index B = input.Size(0);
    Autoalg::Mdarray x_bsd =
        input.View({B, seq_len_, in_per_token_});  // (B, S=28, 28)

    // token projection
    Autoalg::Mdarray x2d = x_bsd.View({B * seq_len_, in_per_token_});  // (B*S, 28)
    Autoalg::Mdarray emb =
        tok_proj_.Forward(x2d).View({B, seq_len_, d_model_});  // (B,S,D)

    // add positional encoding (broadcast on batch)
    Autoalg::Mdarray x = emb + pe_;  // (B,S,D)

    // encoder blocks
    for (auto& blk : blocks_) {
      x = blk->Forward(x);
    }

    // pool over sequence (mean) -> (B, D)
    Autoalg::Mdarray pooled = Autoalg::Operator::CreateOperationMean(x, 1);

    // classifier head -> (B, n_classes)
    Autoalg::Mdarray logits = head_.Forward(pooled);
    return logits;
  }

  Autoalg::Learning::ParamsDict Parameters() override {
    Autoalg::Learning::ParamsDict dict = {
        {"tok_proj", tok_proj_.Parameters()},
        {"head", head_.Parameters()},
        {"block_1", blocks_[0]->Parameters()},
        {"block_2", blocks_[1]->Parameters()},
    };
    return dict;
  }

 private:
  Autoalg::Index in_per_token_, seq_len_, d_model_, n_heads_, d_ff_, n_layers_;
  Autoalg::Learning::Linear tok_proj_;
  Autoalg::Learning::Linear head_;
  std::vector<std::unique_ptr<TransformerEncoderBlock>> blocks_;

  // positional encoding buffer & view
  std::vector<Autoalg::BasicData> pe_buf_;
  Autoalg::Mdarray pe_;
};

// ====== Training (对齐你的 MLP 例子) ======
int main() {
  // 可调超参
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

  // model & criterion
  // rows-as-tokens: 28 tokens，每个 token 维度 28 -> 映射到 d_model
  TransformerMNIST model(/*in_per_token=*/28, /*seq_len=*/28,
                         /*d_model=*/64, /*n_heads=*/4,
                         /*d_ff=*/128, /*n_layers=*/2,
                         /*n_classes=*/10);
  Autoalg::Learning::CrossEntropy criterion;

  // optimizer
  Autoalg::Learning::StochasticGradientDescentWithMomentum optimizer(
      model.Parameters(), lr, momentum);

  Autoalg::Index n_samples;
  const Autoalg::BasicData* batch_samples;
  const Autoalg::Index* batch_labels;

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

      // 输入与 MLP 相同：(B, 784)
      Autoalg::Mdarray input(batch_samples,
                        {n_samples, Autoalg::SourceData::MNIST::Img::pixels_size_});

      Autoalg::Mdarray logits = model.Forward(input);
      Autoalg::Mdarray loss = criterion.Forward(logits, batch_labels);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      if (j % print_iterators == 0) {
        LOG_MDA_INFO("iter " << j << " | loss: " << loss.Item())
      }
    }

    // eval
    LOG_MDA_INFO("Epoch " << i << " Evaluating...")
    Autoalg::Index total_samples = 0, correct_samples = 0;
    for (Autoalg::Index j = 0; j < val_dataset.BatchesSize(); ++j) {
      std::tie(n_samples, batch_samples, batch_labels) =
          val_dataset.GetBatch(j);
      Autoalg::Mdarray input(batch_samples,
                        {n_samples, Autoalg::SourceData::MNIST::Img::pixels_size_});

      Autoalg::Mdarray logits = model.Forward(input);
      Autoalg::Mdarray predict = Autoalg::Operator::CreateOperationArgmax(logits, 1);
      for (Autoalg::Index k = 0; k < n_samples; ++k) {
        ++total_samples;
        Autoalg::Index pd_label = static_cast<Autoalg::Index>(predict[{k}]);
        if (pd_label == batch_labels[k]) ++correct_samples;
      }
    }
    LOG_MDA_INFO("total samples: " << total_samples << " | correct samples: "
                                   << correct_samples << " | acc: ")
    LOG_MDA_INFO(static_cast<Autoalg::BasicData>(correct_samples) / total_samples)
  }

  steady_clock::time_point end_tp = steady_clock::now();
  std::chrono::duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  LOG_MDA_INFO("Training finished. Training took " << time_span.count()
                                                   << " seconds.")
  return 0;
}
