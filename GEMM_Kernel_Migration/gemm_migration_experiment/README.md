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
- `prompts/`: the active prompt set is capped at four baseline prompts: P0 no
  hardware hint, P1 target name only, P2 hardware feature table, and P3 target
  example. Earlier exploratory prompts are preserved in Git history and
  historical reports, not in the current prompt directory or default matrix.
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
  --model azure/gpt-5.4 \
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
  evaluated on H100. T1 V100-to-A100 and T3 H100-to-A100 have now been built,
  correctness-tested, irregular-audited, audit-pass formal-timed, and SASS
  scanned on A100. V100 target tasks remain generated but not target-evaluated.
- `20260524_run02_gpt54`: `gpt-5.4`, P0-P3 baseline. This is the primary H100
  prompt-ablation run for answering whether no hint, target name, hardware
  feature table, or target example helps.
- `20260524_run03_gpt54_promptfix` and `20260524_run04_wiki_gpt54_large` are
  exploratory follow-up runs. They are useful evidence about failure modes, but
  they are no longer part of the active prompt matrix.
- Earlier smoke timing CSVs are preserved as `performance_smoke_results.csv`;
  `performance_results.csv` now contains formal timing for the H100 runs above.
- A stricter irregular-shape audit suite is available at
  `configs/shapes_correctness_audit.json`. It exposes boundary/alignment issues
  missed by the original aligned correctness suite: run01 has 4/10 compiled T2
  candidates passing the audit, run02 has 3/6, run03 has 3/3, and run04 has
  20/33 audited compiled candidates. In run04, P4 has 9/10 aligned correctness
  but 0/10 irregular-audit pass, so aligned correctness alone is not sufficient.
- In the primary `gpt-5.4` P0-P3 baseline, only P3 target-example prompting has
  robust H100 evidence: 3/3 compile, 3/3 aligned correctness, 3/3 irregular
  audit pass, and best audit-pass formal timing of 6.625 TFLOPS. P0 and P1 do
  not pass irregular audit; P2 has 0/3 compile in this run.
- Nsight Compute was attempted for the top candidate in each H100 run, but the
  current node lacks the required `nsight-compute` installation directory behind
  the exposed `ncu` wrapper, so profiler evidence is unavailable there for now.
- `reports/20260524_baseline_p0_p3/final_report.md` summarizes the P0-P3 H100
  baseline prompt ablation.

The run01 A100 target tasks are complete: `T1_v100_to_a100` and
`T3_h100_to_a100` were built with `--arch sm80` and evaluated with the aligned
suite, irregular audit, audit-pass performance, static feature scan, and SASS
scan. The remaining hardware gap is V100 target evaluation for
`T9_a100_to_v100` and `T10_h100_to_v100`.

## Interpreting Results

The generated report answers the five workflow questions by separating:

- why each GEMM implementation style fits V100, A100, or H100;
- why the same implementation can become invalid or suboptimal on another GPU;
- which migration patterns naive LLMs can reliably produce versus only name or
  imitate superficially;
- compile/run/correctness/performance evidence;
- aligned-shape correctness versus irregular-shape boundary correctness;
- source-code feature claims;
- SASS-level confirmation;
- optional Nsight Compute evidence.

The final judgement should prioritize target-GPU execution results over textual
claims in the LLM response. Rows with `compile_status=not_run` are generated
cases that have not yet been evaluated on their target GPU, not compile
failures.
