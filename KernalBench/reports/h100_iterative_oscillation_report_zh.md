# KernelBench H100 迭代复现震荡报告

## 1. 实验目的

这份报告关注的是 KernelBench 复现过程中的一个具体问题：同一个算子经过多轮 LLM 生成、反馈、再生成之后，性能是否会稳定提升，还是会出现退化和震荡。

这里的“震荡”不是单纯指运行时间测量有噪声，而是指迭代过程中模型生成的 kernel 质量不稳定。例如：

- 第 0 轮正确且较快，第 1 轮变慢，第 2 轮又变快。
- 第 0 轮正确，第 1 轮错误或不满足约束，第 2 轮又恢复正确。
- feedback 本来应该让模型改进，但实际诱导模型走向 PyTorch fallback、错误答案或非法 CUDA 实现。

本报告使用已经完成的 H100 Level 1 小样本实验结果，重点回答四个问题：

1. 性能是否随着迭代单调提升？
2. 不同算子的收敛行为是否不同？
3. 哪些算子出现了震荡或退化？
4. 哪些算子的优化更依赖迭代？

## 2. 实验设置

硬件环境：远端 H100，`NVIDIA H100 80GB HBM3`，CUDA toolkit 使用 `/usr/local/cuda-12.9`。

模型与任务设置：

- 生成模型：Azure OpenAI `gpt-5.2-chat`
- KernelBench level：Level 1
- backend：CUDA
- precision：FP32
- 每个问题迭代轮数：3 轮
- 正确性 trials：2
- 性能 trials：10
- timing method：`cuda_event`

本次主要使用三组 run：

| run | 用途 | 问题集合 |
| --- | --- | --- |
| `h100_iter_smoke_level1` | 第一版迭代实验，观察整体趋势 | P1, P6, P19, P33, P41 |
| `h100_iter_strict_level1` | 收紧 prompt 后复跑，观察约束漂移是否改善 | P1, P6, P19, P33, P41 |
| `h100_iter_isolated_norm_pool` | 对 BatchNorm / MaxPool 使用子进程隔离评测，避免坏 CUDA kernel 污染后续评测 | P33, P41 |

这里的 speedup 定义为：

```text
speedup = PyTorch reference runtime / generated kernel runtime
```

所以 speedup 大于 1 表示生成 kernel 比 PyTorch reference 更快；speedup 小于 1 表示生成 kernel 更慢。

## 3. 核心结论

这次小样本复现说明，KernelBench 的迭代优化不是稳定单调过程。不同算子表现差异很大：

- Matmul 类问题基本没有体现出正向迭代收益，后续轮次常出现约束漂移或错误答案。
- ReLU 这类简单 activation 在收紧 prompt 后可以稳定小幅提升，属于早期收敛型。
- BatchNorm 正确性最不稳定，迭代有时能找到正确 kernel，但下一轮又可能变成错误答案。
- MaxPool1D 是最明显的迭代相关算子之一。第一版 run 中 3 轮持续提升；隔离复跑后则出现“快、慢、再变快”的震荡曲线。

因此，不能简单认为“多迭代几轮一定更好”。更准确的结论是：迭代对部分算子有帮助，但会引入明显的不稳定性；是否值得迭代取决于算子类型和评测隔离策略。

## 4. 第一版 run：`h100_iter_smoke_level1`

第一版 run 覆盖 5 个 Level 1 算子，分别代表 matmul、activation、normalization、pooling。

| problem | operator | iter 0 | iter 1 | iter 2 | 行为分类 |
| --- | --- | ---: | ---: | ---: | --- |
| P1 Square matmul | matmul | 0.158 | fail | fail | 低迭代收益，后续约束漂移 |
| P6 large-K matmul | matmul | 0.061 | fail | fail | 低迭代收益，后续约束漂移 |
| P19 ReLU | activation | 0.588 | fail | fail | 后续约束漂移，第一版 prompt 不稳定 |
| P33 BatchNorm | normalization | 0.521 | 0.754 | fail | 第 1 轮明显提升，第 2 轮退化 |
| P41 MaxPool1D | pooling | 1.390 | 1.423 | 2.134 | 逐轮提升，明显依赖迭代 |

第一版 run 给出两个重要信号。

第一，MaxPool1D 在三轮中持续提升，best speedup 出现在第 2 轮，从 1.390 提升到 2.134，相对第一轮提升约 53.6%。这说明 pooling 类算子存在可以通过反馈迭代逐步发现的优化空间。

第二，matmul 和 ReLU 的失败主要不是 CUDA 编译失败，而是 static checker 拒绝。抽查生成代码后发现，模型在 feedback 后倾向于改成 `torch.matmul`、ATen/cuBLAS wrapper、`torch.relu_` 这类 forbidden fallback。也就是说，迭代反馈让模型“变聪明地绕过任务”，而不是继续写合法 custom kernel。

## 5. 收紧 prompt 后：`h100_iter_strict_level1`

为了减少 forbidden fallback，我收紧了 feedback prompt，明确禁止 PyTorch/ATen/cuBLAS/cuDNN fallback，并要求 CUDA backend 必须包含真实 `__global__` kernel。

| problem | operator | iter 0 | iter 1 | iter 2 | 行为分类 |
| --- | --- | ---: | ---: | ---: | --- |
| P1 Square matmul | matmul | 0.174 | wrong | 0.157 | 正确性震荡，最终速度低于第 0 轮 |
| P6 large-K matmul | matmul | fail | 0.034 | wrong | 可行点不稳定 |
| P19 ReLU | activation | 0.935 | 0.961 | 1.000 | 稳定小幅提升，早期收敛 |
| P33 BatchNorm | normalization | eval exception | eval exception | eval exception | 被 CUDA context failure 污染，不能直接作为算子结论 |
| P41 MaxPool1D | pooling | eval exception | eval exception | eval exception | 被前序 CUDA failure 污染，不能直接作为算子结论 |

这组 run 说明，收紧 prompt 对 ReLU 有明显帮助。P19 从 0.935 到 1.000，三轮都正确，没有约束漂移。

但 matmul 仍然不稳定。P1 第 0 轮正确，第 1 轮错误，第 2 轮恢复正确但速度从 0.174 下降到 0.157；这属于正确性和性能共同退化。P6 只在第 1 轮找到一个正确 kernel，但速度很差，后续又错误。

更关键的是，P33 的一个坏 CUDA kernel 触发了 `unspecified launch failure`，之后同一个 Python 进程里的 CUDA context 被污染，导致 P33 后续轮次和 P41 都出现 `eval_exception`。这说明做震荡研究时，必须隔离每次评测，否则一个坏 kernel 会让后续样本出现假失败。

## 6. 隔离评测后：`h100_iter_isolated_norm_pool`

为了解决 CUDA context 污染，我给 runner 加了 `--isolate-eval`，让每个候选 kernel 在独立子进程里评测。然后只重跑 BatchNorm 和 MaxPool。

| problem | operator | iter 0 | iter 1 | iter 2 | 行为分类 |
| --- | --- | ---: | ---: | ---: | --- |
| P33 BatchNorm | normalization | wrong | 0.599 | wrong | 正确性反复，可行点不稳定 |
| P41 MaxPool1D | pooling | 1.517 | 1.365 | 1.471 | 性能震荡，全部正确 |

这组结果是目前最干净的震荡证据。

P33 BatchNorm 三轮都能编译，但第 0 轮和第 2 轮是 wrong answer，只有第 1 轮正确。这说明 BatchNorm 的难点首先是正确性，而不是单纯性能优化。它对迭代有依赖，因为第 1 轮能从错误中找到可行解；但它也高度不稳定，因为下一轮又退回错误解。

P41 MaxPool1D 三轮都正确，但速度不是单调提升：

```text
iter 0: speedup 1.517, runtime 7.12 us
iter 1: speedup 1.365, runtime 7.91 us
iter 2: speedup 1.471, runtime 7.34 us
```

这就是典型性能震荡：第 1 轮比第 0 轮差，第 2 轮又恢复一部分，但没有超过第 0 轮。它和第一版 run 的 `1.390 -> 1.423 -> 2.134` 一起说明 MaxPool 是迭代敏感算子，但反馈优化方向不稳定。

## 7. 按算子类型总结收敛行为

### Matmul：低迭代收益，容易退化

P1 和 P6 的共同特点是 reference 已经很强，模型写 custom CUDA 很难超过 PyTorch/cuBLAS 路径。第一版 run 中，feedback 后模型直接漂到 forbidden fallback；收紧 prompt 后，fallback 减少了，但 wrong answer 和性能退化仍然存在。

对 matmul 来说，简单的自然语言 feedback 不足以稳定提高性能。它可能需要专门模板、CUTLASS/Triton 风格约束，或者更强的 shape-specific blocking 指导。

### Activation：早期收敛，小幅收益

P19 ReLU 在第一版 run 中出现 constraint violation，但收紧 prompt 后三轮都正确，并从 0.935 提升到 1.000。这个算子比较简单，模型容易找到接近 reference 的实现。

这类算子不太需要长迭代。多轮迭代可能带来小幅收益，但边际收益不大。

### Normalization：正确性不稳定

P33 BatchNorm 的行为最像“可行性搜索”。隔离评测后，第 1 轮能找到正确 kernel，但第 0 和第 2 轮都是 wrong answer。

这说明 normalization 类算子更依赖迭代来找到正确实现，但也更容易在后续优化中破坏正确性。对这类算子，反馈 prompt 应该更强调保持数值语义，而不是只追求 runtime。

### Pooling：迭代敏感，存在真实性能震荡

P41 MaxPool1D 是当前最有研究价值的例子。它在第一版 run 中表现为强迭代收益，在隔离 run 中表现为非单调震荡。

这说明 pooling 类算子的搜索空间比较适合 LLM 通过迭代探索，例如展开 kernel、调整边界判断、减少动态控制流等。但这种探索不是稳定爬坡，而是会在不同实现之间来回摆动。

## 8. 震荡来源分析

目前观察到的震荡至少有四类来源：

1. 约束漂移：模型为了“优化”改用 PyTorch/ATen/cuBLAS/cuDNN fallback，被 static checker 拒绝。第一版 matmul 和 ReLU 主要是这种情况。
2. 正确性震荡：模型生成的 kernel 能编译，但数值错误。P1、P6、P33 都出现过。
3. 性能震荡：所有轮次都正确，但 runtime 先变慢再变快。隔离 run 的 P41 是最清楚例子。
4. 评测污染：坏 CUDA kernel 触发 launch failure 后污染同进程 CUDA context，导致后续样本假失败。`h100_iter_strict_level1` 的 P33/P41 就受到了这个问题影响。

因此，后续做正式统计时，应把这四类分开。否则会把 prompt 约束问题、kernel 正确性问题、真实性能震荡和评测系统污染混在一起。

## 9. 对正式复现实验的建议

1. 默认启用 `--isolate-eval`。这是研究震荡行为的必要条件，否则坏 kernel 会污染后续结果。
2. 每个算子至少跑 5 轮，最好每类算子选 5 到 10 个问题。3 轮只能看到初步趋势，不足以稳定估计收敛分布。
3. 报告里同时保留 `speedup` 和 `best_speedup_so_far`。前者看震荡，后者看迭代搜索是否找到过更好解。
4. 把失败模式单独统计：`constraint_violation`、`wrong_answer`、`compile_error`、`eval_exception` 不应该混成一个 fail。
5. 对 normalization/reduction/pooling 这类语义复杂算子，在 feedback 中优先强调正确性保持；对 matmul 这类高性能算子，应使用更强的 kernel 模板或 backend-specific 示例。

## 10. 当前阶段结论

基于 H100 Level 1 小样本结果，KernelBench 迭代优化存在明显的算子差异：

- 最依赖迭代：MaxPool1D、BatchNorm。
- 最容易早期收敛：ReLU。
- 最容易低收益或退化：matmul。
- 最明确的真实性能震荡：隔离评测下的 MaxPool1D，speedup `1.517 -> 1.365 -> 1.471`。
- 最明显的正确性震荡：BatchNorm，`wrong -> correct -> wrong`。

因此，这次复现的核心发现不是“迭代越多越快”，而是“迭代像搜索过程”。它能偶尔找到更优 kernel，但也会偏离约束、破坏正确性或在多个局部实现之间震荡。后续正式实验应以 best-so-far 作为优化收益指标，同时单独报告逐轮 speedup 曲线来刻画震荡行为。

## 11. 结果取得过程校验

本节是对上述结果的事后审计，只读取已经保存的产物，不重新生成、不重新评测。

### 11.1 产物完整性

三组 run 的核心产物均存在：

- `runs/<run_name>/iteration_config.json`
- `runs/<run_name>/iteration_results.json`
- `runs/<run_name>/iteration_results.jsonl`
- 每一轮的 prompt、raw output、kernel 文件
- `reports/<run_name>/iteration_metrics.csv`
- `reports/<run_name>/convergence_summary.csv`
- speedup/runtime/best-so-far 相关图像

隔离评测 run `h100_iter_isolated_norm_pool` 还保存了每次子进程评测的：

- `eval_request.json`
- `eval_result.json`
- `eval_stdout.txt`
- `eval_stderr.txt`

这些子进程请求均使用 `device=cuda:0`、`backend=cuda`、`num_correct_trials=2`、`num_perf_trials=10`。

### 11.2 配置一致性

三组 run 的配置和报告描述一致：

| run | records | expected records | problem ids | iterations | isolated |
| --- | ---: | ---: | --- | ---: | --- |
| `h100_iter_smoke_level1` | 15 | 15 | 1, 6, 19, 33, 41 | 3 | no |
| `h100_iter_strict_level1` | 15 | 15 | 1, 6, 19, 33, 41 | 3 | no |
| `h100_iter_isolated_norm_pool` | 6 | 6 | 33, 41 | 3 | yes |

保存的评测 metadata 中，三组 run 均记录了硬件为 `NVIDIA H100 80GB HBM3`，device 为 `cuda:0`。

### 11.3 数字一致性

重新从 `iteration_results.json` 计算以下字段，均与 CSV 汇总一致：

- `speedup = ref_runtime / runtime`
- `best_speedup_so_far`
- `compile_rate`
- `correctness_rate`
- `constraint_violation_count`
- `best_iter`
- `best_speedup`
- `failure_mode`

同时，`iteration_results.json` 和 `iteration_results.jsonl` 内容完全一致；没有发现 JSON/JSONL/CSV 之间的数字错配。

### 11.4 关键结论证据

关于约束漂移：第一版 run 的失败样本确实包含 `torch.matmul`、`at::matmul`、`torch.relu_`、`at::batch_norm` 等 fallback 或 wrapper 路径；对应 metadata 中也有 `Missing __global__ kernel definition`、`Uses torch computation op` 等 static checker 信息。

关于 prompt 收紧：`h100_iter_strict_level1` 和 `h100_iter_isolated_norm_pool` 的反馈轮次 prompt 文件中确实包含禁止 PyTorch/ATen/cuBLAS/cuDNN fallback 的规则，并要求 CUDA backend 包含真实 `__global__` kernel。

关于 CUDA context 污染：`h100_iter_strict_level1` 中，P33 第 0 轮先在 `torch.cuda.synchronize` 阶段触发 `CUDA error: unspecified launch failure`；后续 P33/P41 多个样本在 `torch.cuda.manual_seed_all` 阶段继续报同样的 launch failure。这支持“同进程 CUDA context 被坏 kernel 污染”的判断。因此，strict run 中 P33/P41 的 `eval_exception` 不应被当作这些算子自身的独立迭代结果。

关于隔离评测：`h100_iter_isolated_norm_pool` 的每个候选 kernel 都有独立子进程的 request/result 文件。隔离后，P41 三轮均正确，P33 三轮均编译但只有第 1 轮正确，这与报告中的震荡描述一致。

### 11.5 仍需保留的限制

本报告的结论是小样本、初步复现结论，不应过度外推。主要限制如下：

1. 样本量很小：主要只有 5 个 Level 1 问题，每个 3 轮；隔离评测只覆盖 P33/P41。
2. trials 较少：正确性 trials 为 2，性能 trials 为 10；小幅 speedup 变化可能包含计时噪声。
3. reference runtime 每轮都会重新测量，因此 speedup 的变化同时受到 generated kernel runtime 和 reference runtime 测量波动影响；判断震荡时应同时看 `runtime_us`。
4. strict run 不是隔离评测，因此 P33/P41 在 strict run 中的失败只能作为“评测污染”证据，不能作为 normalization/pooling 的独立性能结论。
5. LLM 生成过程没有固定随机种子或可复现采样记录，因此保存的 prompt/raw/kernel/result 文件是本次 run 的证据，但重新运行不保证生成完全相同代码。

校验后的结论是：报告中的主要数字和现象都能从保存产物中追溯，未发现计算错误或文件错配；但结论应明确限定为 H100 Level 1 小样本的初步观察，正式结论需要更多问题、更多迭代轮次和默认隔离评测。