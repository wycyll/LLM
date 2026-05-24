# GEMM Kernel Migration Experiment

This directory turns `../workflow.md` into a runnable experiment harness for
studying naive LLM migration of NVIDIA FP16 GEMM kernels across V100, A100, and
H100.

The experiment keeps one fixed launcher ABI for every generated kernel:

```cpp
extern "C" void launch_gemm_kernel(
    const half* A,
    const half* B,
    half* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
);
```

Formal results must be collected on the target GPU named by each task. A local
smoke run on another GPU is only a build-path check and must not be used as a
V100/A100/H100 migration conclusion.

## What Is Included

- `configs/`: GPU metadata, shape suites, model defaults, and task matrix.
- `prompts/`: P0-P7 ablation templates from the workflow and follow-up prompt
  fixes. P4 is compile/correctness constrained for H100; P5 is a Hopper feature
  probe for WGMMA/TMA/mbarrier/warpgroup claims. P6/P7 are KernelWiki-informed
  Hopper prompts that explicitly avoid Blackwell-only features.
- `references/`: source seed kernels and architecture notes for V100, A100, and
  H100. These are useful inputs for generation and harness smoke tests; replace
  or augment them with expert kernels before reporting native-reference
  performance.
- `harness/`: CMake CUDA runner that compiles one candidate kernel, compares it
  against cuBLAS, and reports correctness/performance as JSONL.
- `scripts/`: end-to-end automation for case preparation, LLM generation,
  extraction, build, correctness, performance, static/SASS feature checks,
  Nsight Compute profiling, result collection, and report generation.

## Minimal Smoke Run

From this directory:

```bash
python3 scripts/03_build_kernel.py \
  --run_id smoke_local \
  --task smoke_reference \
  --prompt p0_no_hw_hint \
  --sample sample_00 \
  --target_gpu local \
  --arch sm89 \
  --kernel_source references/v100_sm70/kernel.cu

python3 scripts/04_run_correctness.py \
  --run_id smoke_local \
  --target_gpu local \
  --shapes configs/shapes_correctness.json

python3 scripts/05_run_performance.py \
  --run_id smoke_local \
  --target_gpu local \
  --shapes configs/shapes_performance.json \
  --warmup 2 \
  --repeat 5 \
  --include_partial
```

Use `--arch sm80` on A100, `--arch sm90` on H100, and `--arch sm70` on V100 for
formal experiment builds.

## Formal Run Outline

```bash
python3 scripts/00_prepare_cases.py --run_id 20260524_run01 --samples 3

python3 scripts/01_generate_llm.py \
  --run_id 20260524_run01 \
  --model azure/gpt-5.2-chat \
  --env_file ../../KernalBench/.env \
  --all_cases

python3 scripts/02_extract_code.py --run_id 20260524_run01

# Run these on the target GPU nodes:
python3 scripts/03_build_kernel.py --run_id 20260524_run01 --target_gpu a100 --arch sm80
python3 scripts/03_build_kernel.py --run_id 20260524_run01 --target_gpu h100 --arch sm90
python3 scripts/04_run_correctness.py --run_id 20260524_run01 --target_gpu a100
python3 scripts/04_run_correctness.py --run_id 20260524_run01 --target_gpu h100
python3 scripts/11_run_correctness_audit.py --run_id 20260524_run01 --target_gpu a100
python3 scripts/11_run_correctness_audit.py --run_id 20260524_run01 --target_gpu h100
python3 scripts/05_run_performance.py --run_id 20260524_run01 --target_gpu a100
python3 scripts/05_run_performance.py --run_id 20260524_run01 --target_gpu h100

python3 scripts/06_static_feature_check.py --run_id 20260524_run01
python3 scripts/07_sass_feature_check.py --run_id 20260524_run01
python3 scripts/08_profile_ncu.py --run_id 20260524_run01 --selected top_correct_kernels
python3 scripts/09_collect_results.py --run_id 20260524_run01
python3 scripts/10_make_report.py --run_id 20260524_run01
```

On the current H100 SSH node discovered during setup, CUDA tools are installed
but not on the default PATH, and `/usr/local/bin/cmake` is broken. Before
running build/profile commands there, use the CMake/Ninja from the `flux` conda
environment first:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="/root/miniconda3/envs/flux/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

## Current Run Status

- `20260524_run01`: `gpt-5.2-chat`, P0-P3 baseline. T2 A100-to-H100 has been
  built, correctness-tested, formal-timed (`warmup=10`, `repeat=100`), and SASS
  scanned on H100. A100/V100 target tasks are generated but not target-evaluated.
- `20260524_run02_gpt54`: `gpt-5.4`, P0-P3 baseline. T2 A100-to-H100 has the
  same H100 evaluation coverage as run01.
- `20260524_run03_gpt54_promptfix`: `gpt-5.4`, P4-P5 prompt ablation. T2
  A100-to-H100 has the same H100 evaluation coverage as run01.
- `20260524_run04_wiki_gpt54_large`: `gpt-5.4`, P3/P4/P6/P7 with 10 samples
  per prompt. T2 A100-to-H100 has H100 compile, aligned correctness, irregular
  audit, audit-pass-only formal timing, static feature scan, and SASS scan.
- Earlier smoke timing CSVs are preserved as `performance_smoke_results.csv`;
  `performance_results.csv` now contains formal timing for the H100 runs above.
- A stricter irregular-shape audit suite is available at
  `configs/shapes_correctness_audit.json`. It exposes boundary/alignment issues
  missed by the original aligned correctness suite: run01 has 4/10 compiled T2
  candidates passing the audit, run02 has 3/6, run03 has 3/3, and run04 has
  20/33 audited compiled candidates. In run04, P4 has 9/10 aligned correctness
  but 0/10 irregular-audit pass, so aligned correctness alone is not sufficient.
- In run04, P6 is the strongest current H100 prompt: 8/10 compile, 6/10 aligned
  all-correct, 6/7 audited compiled candidates passing irregular shapes, and
  best audit-pass formal timing of 30.359 TFLOPS on the performance suite.
- Nsight Compute was attempted for the top candidate in each H100 run, but the
  current node lacks the required `nsight-compute` installation directory behind
  the exposed `ncu` wrapper, so profiler evidence is unavailable there for now.
- `results/model_comparison_20260524_t2.md` summarizes the current H100 model
  and prompt comparison.

When an A100 is available, the next formal target evaluations are the generated
run01 A100 tasks: `T1_v100_to_a100` and `T3_h100_to_a100`, built with `--arch
sm80` and evaluated with the aligned suite, irregular audit, performance, static
feature scan, and SASS scan.

## Interpreting Results

The generated report answers the five workflow questions by separating:

- compile/run/correctness/performance evidence;
- aligned-shape correctness versus irregular-shape boundary correctness;
- source-code feature claims;
- SASS-level confirmation;
- optional Nsight Compute evidence.

The final judgement should prioritize target-GPU execution results over textual
claims in the LLM response. Rows with `compile_status=not_run` are generated
cases that have not yet been evaluated on their target GPU, not compile
failures.
