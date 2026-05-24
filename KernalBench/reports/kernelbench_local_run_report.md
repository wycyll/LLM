# KernelBench 本地运行报告

日期：2026-05-20  
项目：ScalingIntelligence/KernelBench  
本地目录：`/home/jenny/LLM/KernalBench`

## 1. 运行目的

本次工作的目标是验证 KernelBench 是否能够在当前本地机器上跑通，并确认其 benchmark 流程是否可用。KernelBench 的核心用途是评测 LLM 生成 GPU kernel 的能力，主要关注两个指标：

- 正确性：生成的 kernel 输出是否与 PyTorch reference 一致。
- 性能：生成的 kernel 相比 PyTorch reference 是否有加速。

本次验证覆盖了三条路径：

- 官方单题本地评测脚本 `scripts/run_and_check.py`。
- 官方批量评测脚本 `scripts/eval_from_generations.py`。
- 官方汇总分析脚本 `scripts/benchmark_eval_analysis.py`。

## 2. 本地环境

当前机器可以运行 KernelBench 的 CUDA 本地评测链路，关键配置如下：

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8 GB |
| NVIDIA Driver | 581.83 |
| CUDA Toolkit | 12.9 |
| Python | 3.10.20，项目虚拟环境 `.venv` |
| PyTorch | 2.5.1+cu124 |
| Triton | 3.1.0 |
| CUDA 编译器 | `nvcc` 12.9 |
| C/C++ 编译器 | gcc/g++ 13.3 |
| 构建工具 | ninja 1.13.0 |

运行 CUDA extension 时使用的关键环境变量包括：

```bash
PYTHONPATH=/home/jenny/LLM/KernalBench/src
CUDA_HOME=/usr/local/cuda-12.9
PATH=/home/jenny/LLM/KernalBench/.venv/bin:/usr/local/cuda-12.9/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
TORCH_CUDA_ARCH_LIST=8.9
MAX_JOBS=2
```

## 3. 环境准备过程

最初直接 `git clone https://github.com/ScalingIntelligence/KernelBench` 超时，因此改用 GitHub codeload zip 下载并解压到本地目录。

项目官方推荐使用 `uv sync`，但完整同步会下载大量依赖，在当前网络下速度较慢。因此实际采用了最小可运行安装策略：

- 使用 Python 3.10 项目虚拟环境。
- 从 PyTorch CUDA 12.4 wheel 源安装 `torch==2.5.1+cu124`。
- 补充 PyTorch CUDA 运行库：`nvidia-cudnn-cu12`、`nvidia-nccl-cu12`、`nvidia-cuda-cupti-cu12`、`nvidia-nvtx-cu12`。
- 安装官方脚本所需依赖：`pydra-config`、`modal`、`numpy`、`tqdm`、`litellm`、`tabulate` 等。
- 将 Triton 固定为 `3.1.0`，以匹配 PyTorch 2.5.1。较新的 Triton 3.7.0 会导致 `torch.compile` 路径导入错误。

## 4. 单题官方评测结果

首先使用官方脚本 `scripts/run_and_check.py` 对仓库内置 add 示例进行评测。

参考实现：`src/kernelbench/prompts/model_ex_add.py`  
生成 kernel：`src/kernelbench/prompts/model_new_ex_add.py`

运行命令摘要：

```bash
python scripts/run_and_check.py \
  ref_origin=local \
  ref_arch_src_path=src/kernelbench/prompts/model_ex_add.py \
  kernel_src_path=src/kernelbench/prompts/model_new_ex_add.py \
  eval_mode=local \
  num_correct_trials=1 \
  num_perf_trials=5 \
  timing_method=cuda_event
```

输出结果：

| 指标 | 结果 |
| --- | --- |
| compiled | True |
| correctness | True |
| Custom Kernel exec time | 0.0135 ms |
| PyTorch Reference Eager exec time | 0.00942 ms |
| PyTorch Reference torch.compile time | 0.0084 ms |
| Speedup over eager | 0.70x |
| Speedup over torch.compile | 0.62x |

结论：add 示例的自定义 CUDA kernel 能够正确编译和运行，但由于问题规模很小，kernel launch 等开销占比较高，因此速度低于 PyTorch eager 和 torch.compile baseline。

## 5. 官方批量评测闭环

为了验证 KernelBench 的完整 benchmark 后半段流程，本次构造了一个本地 generated sample，并通过官方批量评测和分析脚本执行完整闭环。

选择题目：Level 1 Problem 6  
题目文件：`KernelBench/level1/6_Matmul_with_large_K_dimension_.py`  
任务类型：large-K matrix multiplication

生成样本文件：

```text
runs/local_manual_level1_problem6/level_1_problem_6_sample_0_kernel.py
```

该样本实现 `ModelNew`，使用 `torch.matmul(A, B)` 作为候选 kernel。它不是 LLM 自动生成结果，而是用于验证 KernelBench 批量评测文件格式、执行流程和分析流程是否完整可用。

### 5.1 批量评测

运行脚本：`scripts/eval_from_generations.py`

运行范围：`subset=(6,6)`，即只评测 Level 1 的 Problem 6。

关键参数：

| 参数 | 值 |
| --- | --- |
| dataset_src | local |
| level | 1 |
| subset | (6, 6) |
| num_correct_trials | 1 |
| num_perf_trials | 5 |
| num_gpu_devices | 1 |
| timing_method | cuda_event |

输出文件：

```text
runs/local_manual_level1_problem6/eval_results.json
```

评测结果：

| 指标 | 结果 |
| --- | --- |
| problem_id | 6 |
| sample_id | 0 |
| compiled | true |
| correctness | true |
| correctness_trials | 1 / 1 |
| candidate runtime mean | 12.6 ms |
| candidate runtime std | 0.434 ms |
| candidate runtime min | 12.0 ms |
| candidate runtime max | 13.3 ms |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

评测过程中脚本还打印了 reference runtime，均值约为 `14.2 ms`。

### 5.2 本地 baseline

为了运行官方分析脚本，创建了本地 RTX 4060 smoke baseline：

```text
results/timing/local_rtx4060_smoke/baseline_time_torch_problem6.json
```

其中 Problem 6 的 reference mean runtime 记录为：

```text
14.2 ms
```

### 5.3 汇总分析

运行脚本：`scripts/benchmark_eval_analysis.py`

输出文件：

```text
runs/local_manual_level1_problem6/analysis_summary.json
```

分析结果：

| 指标 | 结果 |
| --- | --- |
| run_name | local_manual_level1_problem6 |
| level | 1 |
| total_count | 100 |
| total_eval | 1 |
| compiled_count | 1 |
| correct_count | 1 |
| compilation_rate | 0.01 |
| correctness_rate | 0.01 |
| geo_mean_speedup | 1.126984126984127 |

Fast_p 结果：

| Speedup threshold | Fast_p score |
| --- | --- |
| 0.0 | 1.0 |
| 0.5 | 1.0 |
| 0.8 | 1.0 |
| 1.0 | 1.0 |
| 1.5 | 0.0 |
| 2.0 | 0.0 |

解释：`total_count=100` 是因为分析脚本默认把整个 Level 1 的 100 个问题作为总体；本次只评测了 Problem 6，因此 compiled/correctness rate 显示为 1%。这不代表失败，而是本次只运行了一个子集。

## 6. 当前完成度

本次已经跑通的部分：

- CUDA 可用性验证。
- PyTorch GPU 张量计算验证。
- CUDA extension 编译验证。
- 官方 `run_and_check.py` 单题本地评测。
- 官方 `eval_from_generations.py` 批量评测。
- 官方 `benchmark_eval_analysis.py` 汇总分析。
- 本地结果文件和分析 JSON 生成。

尚未完整执行的部分：

- LLM 自动生成 kernels：当前环境未检测到 OpenAI、Gemini、DeepSeek、Anthropic 等常见 API key，因此无法真实调用 `generate_samples.py` 生成模型输出。
- Level 1 全量 100 题本地评测：当前 GPU 为 8GB 显存，Level 1 中部分题目的输入尺寸非常大，直接全量运行很可能 OOM 或耗时过长。
- 完整本机 baseline：目前只为 Problem 6 创建了 smoke baseline。如果要计算完整公平指标，需要为目标 subset 或全量 level 生成对应硬件上的 baseline。

## 7. 结论

KernelBench 在当前机器上已经完成本地运行验证，核心 benchmark 后半段流程可以正常工作。官方单题评测、批量评测和结果分析都已成功执行，并生成了可检查的 JSON 结果。

当前机器适合进行小规模、本地子集级别的 KernelBench 调试和流程验证。若要进行正式、完整的 benchmark，需要补充以下条件：

- 配置可用的 LLM API key，用于运行 `generate_samples.py`。
- 根据 RTX 4060 8GB 显存筛选可运行 subset，或改用更大显存 GPU/Modal 云端 GPU。
- 为目标硬件和目标问题集生成完整 baseline。
- 将 correctness/performance trials 从 smoke test 设置提高到官方默认或实验要求，例如 correctness 5 次、performance 100 次。

## 8. 主要产物

| 文件 | 说明 |
| --- | --- |
| `runs/local_manual_level1_problem6/level_1_problem_6_sample_0_kernel.py` | 本地构造的 generated sample |
| `runs/local_manual_level1_problem6/eval_results.json` | 官方批量评测结果 |
| `results/timing/local_rtx4060_smoke/baseline_time_torch_problem6.json` | 本地 Problem 6 baseline |
| `runs/local_manual_level1_problem6/analysis_summary.json` | 官方分析汇总结果 |

## 9. 追加运行：Level 1 Matmul 子集

在 Problem 6 跑通后，又追加运行了 5 个 Level 1 matmul 类问题，用于验证更多官方 problem 能否在本机 RTX 4060 Laptop GPU 上完成评测。由于当前仍没有 LLM API key，本轮继续使用本地构造的 `ModelNew` 作为 generated sample，重点验证评测链路和本机可运行性。

追加 run 名称：

```text
runs/local_manual_level1_matmul_subset
```

追加评测的问题：

| Problem ID | 文件 | 任务 |
| --- | --- | --- |
| 1 | `KernelBench/level1/1_Square_matrix_multiplication_.py` | square matrix multiplication |
| 14 | `KernelBench/level1/14_Matmul_for_upper_triangular_matrices.py` | upper triangular matmul |
| 15 | `KernelBench/level1/15_Matmul_for_lower_triangular_matrices.py` | lower triangular matmul |
| 16 | `KernelBench/level1/16_Matmul_with_transposed_A.py` | matmul with transposed A |
| 17 | `KernelBench/level1/17_Matmul_with_transposed_B.py` | matmul with transposed B |

运行参数：

| 参数 | 值 |
| --- | --- |
| dataset_src | local |
| level | 1 |
| num_correct_trials | 1 |
| num_perf_trials | 3 |
| num_gpu_devices | 1 |
| timing_method | cuda_event |

追加批量评测结果：

| Problem ID | compiled | correctness | candidate mean runtime | reference mean runtime | speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | true | true | 23.0 ms | 21.1 ms | 0.917x |
| 14 | true | true | 25.0 ms | 19.9 ms | 0.796x |
| 15 | true | true | 24.4 ms | 21.1 ms | 0.865x |
| 16 | true | true | 23.0 ms | 20.6 ms | 0.896x |
| 17 | true | true | 24.3 ms | 21.0 ms | 0.864x |

对应输出文件：

| 文件 | 说明 |
| --- | --- |
| `runs/local_manual_level1_matmul_subset/eval_results.json` | 追加 5 题的官方批量评测结果 |
| `results/timing/local_rtx4060_matmul_subset/baseline_time_torch.json` | 追加 5 题的本地 baseline |
| `runs/local_manual_level1_matmul_subset/analysis_summary.json` | 追加 5 题的官方分析汇总结果 |

追加分析结果：

| 指标 | 结果 |
| --- | --- |
| total_eval | 5 |
| compiled_count | 5 |
| correct_count | 5 |
| compilation_rate | 0.05 |
| correctness_rate | 0.05 |
| geo_mean_speedup | 0.8666077015470598 |

Fast_p 结果：

| Speedup threshold | Fast_p score |
| --- | --- |
| 0.0 | 1.0 |
| 0.5 | 1.0 |
| 0.8 | 0.8 |
| 1.0 | 0.0 |
| 1.5 | 0.0 |
| 2.0 | 0.0 |

解释：这 5 个本地构造 sample 都正确通过评测，但没有超过 PyTorch reference baseline。原因是这些 sample 本质上调用 PyTorch 自身算子，而不是实际手写/LLM 生成的优化 CUDA kernel；因此它们主要用于验证更多 problem 的 KernelBench 评测流程，而不是证明 kernel 优化效果。
