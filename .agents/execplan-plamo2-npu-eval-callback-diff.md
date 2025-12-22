# Fix OpenVINO NPU (PLaMo2) crash and match CPU outputs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

Run `plamo2-1b-alignment-id1397-dpo-auto.gguf` on Windows with the OpenVINO backend targeting the Intel NPU and make it behave like CPU execution:

- No crash when `GGML_OPENVINO_SSM_ONLY=0` (full offload path).
- Intermediate tensors dumped via `llama-eval-callback` match CPU (within reasonable fp16/f32 tolerance).

We validate by dumping intermediate tensors during a single forward pass using `examples/eval-callback/llama-eval-callback.exe` and comparing CPU vs NPU outputs. The user requested `GGML_MAX_DIMS=256`; in this repo `GGML_MAX_DIMS` is also a compile-time macro fixed at 4, so we interpret the request as an environment variable that controls how many scalar values are printed per tensor. Concretely, `GGML_MAX_DIMS=256` means: print the first 256 values (plus a sum) for each selected tensor in a stable order.

The key bug to eliminate is “prefix-dependent corruption”: for OpenVINO+NPU, `LLAMA_EVAL_CALLBACK_TENSOR_PREFIX` used to change the numeric results. That indicates the eval-callback execution strategy itself was destabilizing OpenVINO/NPU, rather than a true operator mismatch.

## Progress

- [x] (2025-12-21 06:00Z) Confirmed repo/builds and model file `plamo2-1b-alignment-id1397-dpo-auto.gguf`.
- [x] (2025-12-21 06:30Z) Added `LLAMA_EVAL_CALLBACK_TENSOR_PREFIX` filtering to `examples/eval-callback/eval-callback.cpp` and `GGML_MAX_DIMS`-driven first-N dumping (`EVALCB_VALS`).
- [x] (2025-12-21 08:00Z) Fixed OpenVINO conversion crashes seen with `GGML_OPENVINO_SSM_ONLY=0` (ROPE `n_dims==0`, VIEW-of-VIEW offset/shape issues), and added targeted tracing for `SET_ROWS`.
- [x] (2025-12-22 00:30Z) Identified and fixed the `GGML_OPENVINO_SSM_ONLY=0` crash subgraph during dump-all: a 1-node `FLASH_ATTN_EXT` graph executed through the OpenVINO naive path was forced to fail; removed the forced failure (`ggml/src/ggml-openvino/utils.cpp`).
- [x] (2025-12-22 01:00Z) Mitigated OpenVINO teardown instability by keeping the `ov::Core` alive for process lifetime (`ggml/src/ggml-openvino/utils.cpp`).
- [x] (2025-12-22 02:30Z) Improved OpenVINO output selection for debugging (exact-match suffix `$`, include VIEW outputs) (`ggml/src/ggml-openvino/ggml-decoder.cpp`, `ggml/src/ggml-openvino/openvino/utils.cpp`).
- [x] (2025-12-22 04:00Z) Fixed eval-callback prefix-dependent corruption on NPU by changing the scheduler’s eval-callback graph-view splitting for the OpenVINO backend and materializing live-outs (`ggml/src/ggml-backend.cpp`).
- [x] (2025-12-22 04:30Z) Verified CPU vs NPU parity for the first 256 values on representative layers (`ffn_residual-0`, `ffn_residual-1`, `ffn_residual-15`) with `GGML_OPENVINO_SSM_ONLY=0`.

## Surprises & Discoveries

- Observation: On OpenVINO+NPU, eval-callback results used to depend on which tensors were requested.
  Evidence: before the scheduler fix, `ffn_residual-0` had `sum=-5.4395` with only `ffn_residual-0$`, but `sum≈16.06` when adding `mamba_conv1d-0$`.

- Observation: This was not a deterministic operator mismatch; it was an eval-callback execution artifact. Large OpenVINO graph views that include `SSM_CONV` and span multiple layers can produce wrong results on NPU unless we enforce stable boundaries and materialize live-outs.
  Evidence: after cutting OpenVINO views at `SSM_CONV` and layer boundaries, the same minimal prefix (`ffn_residual-0$`) matches CPU again.

- Observation: Very small OpenVINO subgraphs (<20 nodes) in the static NPU path are intentionally executed on the CPU plugin via `naive_compute` for stability. This matters when adding more view boundaries.
  Evidence: `ov_graph_compute_static()` checks `is_naive(cgraph)` and falls back.

## Decision Log

- Decision: Use `LLAMA_EVAL_CALLBACK_TENSOR_PREFIX` as the primary mechanism for selecting tensors to dump, rather than dumping every node.
  Rationale: dumping every node forces extremely small OpenVINO graphs and triggers fragile paths; prefix selection is enough to localize issues.
  Date/Author: 2025-12-21 / agent

- Decision: Fix NPU eval-callback correctness by changing scheduler graph-view splitting for the OpenVINO backend (cut at `SSM_CONV` and layer boundaries) and forcing live-outs to materialize.
  Rationale: forcing additional OpenVINO model outputs changed numerics but did not fully restore correctness; stable graph-view boundaries did.
  Date/Author: 2025-12-22 / agent

## Outcomes & Retrospective

- `GGML_OPENVINO_SSM_ONLY=0` no longer crashes on the previously failing subgraph.
- `llama-eval-callback` can now dump `ffn_residual-*` with only an exact-match prefix (e.g. `ffn_residual-0$`) and still get CPU-equivalent values on NPU, without adding “helper” prefixes.
- Remaining differences are within expected fp16/f32 tolerance (sub-millli absolute error on first 256 values in the tested runs).

## Context and Orientation

Key concepts in this repo for this task:

- A “split” (`ggml_backend_sched_split`) is a contiguous subgraph assigned to a specific backend (CPU/OpenVINO/etc). The scheduler executes splits in order.
- In eval-callback mode (`params.cb_eval` set), the scheduler further executes each split as a sequence of “graph views” (`ggml_graph_view`) so it can stop at intermediate tensors and allow the callback to read them.
- The OpenVINO backend only guarantees that model outputs are materialized into ggml tensor buffers. Intermediate tensors that are not outputs may remain stale/uninitialized, depending on the backend implementation.

Key files:

- `ggml/src/ggml-backend.cpp`: `ggml_backend_sched_compute_splits()`; eval-callback view execution and temporary output marking.
- `examples/eval-callback/eval-callback.cpp`: debug callback, prefix filtering, and first-256-value dumping (`GGML_MAX_DIMS`).
- `ggml/src/ggml-openvino/utils.cpp`: OpenVINO compute paths (static NPU vs dynamic), naive fallback behavior, and OpenVINO core lifetime.
- `ggml/src/ggml-openvino/ggml-decoder.cpp`: OpenVINO graph decoder; forced outputs and VIEW output handling.

## Plan of Work

1) Ensure the model can run with `GGML_OPENVINO_SSM_ONLY=0` without crashing by fixing conversion/runtime issues and stabilizing teardown.

2) Make eval-callback outputs stable and comparable between CPU and NPU:

- Provide deterministic dumping (`sum` + first N values) and prefix filtering in `llama-eval-callback`.
- In the backend scheduler’s eval-callback path, for the OpenVINO backend:
  - cut graph views at `GGML_OP_SSM_CONV` outputs (Mamba conv1d) and at layer boundaries (best-effort via tensor name suffix `*-<layer>`),
  - and temporarily mark the requested tensor plus any produced live-outs as outputs so the backend materializes them.

3) Validate parity by comparing CPU vs NPU dumps for representative layer outputs (`ffn_residual-0`, `ffn_residual-1`, `ffn_residual-15`) and ensure a large fraction of the first 256 values match within tolerance.

## Concrete Steps

All commands run from repo root `C:\\Users\\mitmul\\llama.cpp` in PowerShell.

### Environment setup (OpenVINO runtime DLLs)

    $ovLib = (Resolve-Path .venv/Lib/site-packages/openvino/libs).Path
    $env:PATH = "$ovLib;$env:PATH"

### Build

    cmake --build build-cpu --config Release --target llama-eval-callback
    cmake --build build-openvino-relwithdebinfo --config RelWithDebInfo --target llama-eval-callback

### CPU baseline (example: `ffn_residual-0`)

    $env:LLAMA_EVAL_CALLBACK_TENSOR_PREFIX = "ffn_residual-0$"
    $env:GGML_MAX_DIMS = "256"
    cmd /c ".\\build-cpu\\bin\\Release\\llama-eval-callback.exe -m .\\plamo2-1b-alignment-id1397-dpo-auto.gguf -p hi -n 0 -s 1 -t 1 -tb 1 -c 128 --device none -ngl 0 > intermediate-dumps\\cpu_ffn_residual0.txt 2>&1"

### NPU run (same dump selection)

    $env:GGML_OPENVINO_DEVICE = "NPU"
    $env:GGML_OPENVINO_SSM_ONLY = "0"
    $env:LLAMA_EVAL_CALLBACK_TENSOR_PREFIX = "ffn_residual-0$"
    $env:GGML_MAX_DIMS = "256"
    cmd /c ".\\build-openvino-relwithdebinfo\\bin\\RelWithDebInfo\\llama-eval-callback.exe -m .\\plamo2-1b-alignment-id1397-dpo-auto.gguf -p hi -n 0 -s 1 -t 1 -tb 1 -c 128 --device OPENVINO0 -ngl 999 > intermediate-dumps\\npu_ffn_residual0.txt 2>&1"

### Compare first 256 values (quick numeric check)

    @'
    import re

    def read_text(path):
        data = open(path, 'rb').read()
        if data.startswith(b'\xff\xfe') or data.startswith(b'\xfe\xff'):
            return data.decode('utf-16', errors='ignore')
        return data.decode('utf-8', errors='ignore')

    def vals(path, name):
        pat = re.compile(rf"^EVALCB_VALS {re.escape(name)}\s+(.*)$", re.M)
        m = pat.search(read_text(path))
        if not m:
            raise SystemExit(f"missing vals for {name} in {path}")
        return [float(x) for x in m.group(1).split()]

    cpu = r'intermediate-dumps/cpu_ffn_residual0.txt'
    npu = r'intermediate-dumps/npu_ffn_residual0.txt'
    name = 'ffn_residual-0'
    cv = vals(cpu, name)
    nv = vals(npu, name)
    d = [abs(a-b) for a,b in zip(cv, nv)]
    print('max_abs', max(d), 'mean_abs', sum(d)/len(d))
    '@ | python -

## Validation and Acceptance

Acceptance is met when all of the following are true:

- `GGML_OPENVINO_DEVICE=NPU` and `GGML_OPENVINO_SSM_ONLY=0` runs complete and exit with code 0.
- For `LLAMA_EVAL_CALLBACK_TENSOR_PREFIX='ffn_residual-0$'`, NPU output matches CPU (sum and first 256 values within tolerance, e.g. all first-256 diffs <= 1e-3).
- Same holds for a deeper layer (e.g. `ffn_residual-15$`) to ensure there is no layer-to-layer drift.

## Idempotence and Recovery

- The dump commands are safe to re-run; outputs go under `intermediate-dumps\\`.
- If you want to isolate OpenVINO compilation caching effects, set `GGML_OPENVINO_CACHE_DIR` to a fresh empty folder (or unset it).

## Artifacts and Notes

- `intermediate-dumps\\`: CPU/NPU logs (including `EVALCB` and `EVALCB_VALS`) used for comparisons.
