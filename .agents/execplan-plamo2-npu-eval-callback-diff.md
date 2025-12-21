# Compare PLaMo2 CPU vs Intel NPU intermediate outputs (llama-eval-callback)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

We want a reproducible way to dump intermediate tensors from a single forward pass of `plamo2-1b-alignment-id1397-dpo-auto.gguf` on:

1) the CPU backend, and
2) an Intel NPU-backed build (in this repo: the OpenVINO backend configured for `NPU`)

Then we want to compare those dumps and report which transformer layer first diverges (and how).

“Intermediate output” in this plan means tensors that represent layer boundary states (e.g. `ffn_residual-*`) rather than every ggml node, because node-by-node dumping forces the scheduler to execute tiny subgraphs and can break backends that only support larger graph chunks.

## Progress

- [x] (2025-12-21) Located `llama-eval-callback` example and `cb_eval` integration points (`examples/eval-callback/eval-callback.cpp`, `common/common.h`).
- [x] (2025-12-21) Captured a CPU-only full-node dump once as a baseline (`intermediate-dumps/cpu_eval_callback.txt`) using the provided model and prompt.
- [x] (2025-12-21) Verified OpenVINO runtime DLLs are present in `.venv` and can expose `NPU` via Python (`openvino.Core().available_devices`).
- [x] (2025-12-21) Added an env-var controlled tensor-name prefix filter to `llama-eval-callback` to avoid requesting every node (`LLAMA_EVAL_CALLBACK_TENSOR_PREFIX`).
- [x] (2025-12-21) Identified a failure mode: OpenVINO `NPU` path can execute “naive” tiny graphs (e.g. due to scheduler chunking) and the naive decoder path did not populate `m_inputs`, causing `get_input_ggml_tensor missing key ...` errors.
- [x] (2025-12-21) Patched OpenVINO naive decoder path to also populate `m_inputs` for discovered model inputs (`ggml/src/ggml-openvino/ggml-decoder.cpp`).
- [ ] NPU run completes successfully for `plamo2-1b-alignment-id1397-dpo-auto.gguf` (currently crashes/terminates).
- [ ] CPU and NPU layer-boundary dumps captured with the same tensor selection.
- [ ] Automated diff report produced: first mismatching layer and summary stats.

## Surprises & Discoveries

- Observation: PLaMo2 graphs include tensors with zero-sized dimensions (e.g. shapes containing `0`), especially around recurrent state update bookkeeping; CPU handles these, but accelerated backends may choke if such tensors leak into the offloaded subgraph.
  Evidence: `intermediate-dumps/cpu_eval_callback.txt` contains many `VIEW/GET_ROWS/CPY` nodes with `{..., 0, ...}` shapes.
- Observation: OpenVINO `NPU` code path (`ov_graph_compute_static`) does not wrap compile/infer in `try/catch`, so OpenVINO exceptions can crash the process with limited diagnostics.
  Evidence: `llama-eval-callback` and `llama-cli` terminate with exit code `-1073740791` under `GGML_OPENVINO_DEVICE=NPU`.

## Decision Log

- Decision: Add a tensor-name prefix filter to `llama-eval-callback` via an environment variable instead of adding new CLI args.
  Rationale: Keeps the example minimal while enabling backend-friendly chunking; avoids editing shared argument parsing for all examples.
  Date/Author: 2025-12-21 / agent
- Decision: Treat layer-boundary tensors (e.g. `ffn_residual-*`) as the primary comparison artifact, not all nodes.
  Rationale: Requesting every node forces 1-node graph views, which triggers OpenVINO “naive” compute and destabilizes NPU runs.
  Date/Author: 2025-12-21 / agent

## Outcomes & Retrospective

Pending: NPU execution is not yet stable enough to produce dumps; this plan will be updated once the crash root cause is identified and resolved.

## Context and Orientation

Key files:

- `examples/eval-callback/eval-callback.cpp`: example that sets `common_params.cb_eval` to a graph-evaluation callback.
- `common/common.h`: defines `common_params`, including `cb_eval` and `cb_eval_user_data`.
- `src/llama-context.cpp`: wires `llama_context_params.cb_eval` into the ggml backend scheduler.
- `ggml/src/ggml-openvino/utils.cpp`: OpenVINO backend graph execution; selects static path for `NPU`.
- `ggml/src/ggml-openvino/ggml-decoder.cpp`: graph-to-OpenVINO model decoder; has a “naive” constructor used by `naive_compute`.

Important terms:

- “Scheduler eval callback”: a per-node callback invoked by the ggml backend scheduler (`ggml_backend_sched_eval_callback`). If it returns `true` for a node (on the “ask” pass), the scheduler will execute a graph view up to that node and then re-invoke the callback (on the “data” pass) after synchronizing.
- “Naive compute” (OpenVINO): a simplified path used for very small graphs in the OpenVINO NPU static executor. It builds and compiles a tiny OpenVINO model per call. This path is sensitive to missing input bookkeeping.

## Plan of Work

1. Make OpenVINO NPU failures diagnosable.
   In `ggml/src/ggml-openvino/utils.cpp`, wrap `ov_graph_compute_static` and `naive_compute` in `try/catch` similar to `ov_graph_compute_dynamic`, logging the exception message and returning `GGML_STATUS_FAILED` instead of crashing. This should turn the current hard crash into an actionable error message.

2. Add an opt-in “force dynamic on NPU” switch for debugging.
   In `ggml/src/ggml-openvino/utils.cpp`, add an environment variable (e.g. `GGML_OPENVINO_FORCE_DYNAMIC=1`) that forces `ov_graph_compute` to call `ov_graph_compute_dynamic(cgraph, "NPU")` even when `GGML_OPENVINO_DEVICE=NPU`. This is only to obtain successful runs and dumps; it should not be enabled by default.

3. Get a successful NPU run.
   Using `build-openvino` binaries, run `llama-eval-callback` with:

   - `GGML_OPENVINO_DEVICE=NPU`
   - `LLAMA_EVAL_CALLBACK_TENSOR_PREFIX=ffn_residual-` (or another stable layer-boundary tensor set)
   - `--device OPENVINO0 -ngl 999` to ensure the OpenVINO backend is in play

   Iterate until the run completes (even if some OpenVINO subgraphs fall back to CPU due to unsupported ops).

4. Rebuild and capture CPU baseline with the same code version.
   Rebuild `llama-eval-callback` in a CPU-only build directory (or a build configuration where OpenVINO is disabled) from the current working tree, then run it with the same prompt/seed/context/threads and the same `LLAMA_EVAL_CALLBACK_TENSOR_PREFIX`. Save the output under `intermediate-dumps/`.

5. Parse and diff.
   Write a small script (preferably Python, since `.venv` exists) that:
   - extracts each dumped tensor name and its printed `sum = ...` line,
   - groups by layer index extracted from the name suffix (e.g. `ffn_residual-7`),
   - compares CPU vs NPU sums and reports:
     - first mismatching layer,
     - absolute/relative error per layer,
     - whether mismatches grow with depth.

6. Report results.
   Provide the user:
   - paths to the saved CPU/NPU dump files,
   - the first mismatching layer (or confirm all match within tolerance),
   - a short summary of error magnitude and any patterns.

## Concrete Steps

All commands are run from the repo root (`C:\\Users\\mitmul\\llama.cpp`) in PowerShell.

1) Build the OpenVINO `llama-eval-callback`:

    cmake --build build-openvino --config Release --target llama-eval-callback

2) Run NPU dump (requires OpenVINO DLLs on PATH):

    $ovLib = (Resolve-Path .venv/Lib/site-packages/openvino/libs).Path
    $env:PATH = "$ovLib;$env:PATH"
    $env:GGML_OPENVINO_DEVICE = "NPU"
    $env:LLAMA_EVAL_CALLBACK_TENSOR_PREFIX = "ffn_residual-"
    ./build-openvino/bin/Release/llama-eval-callback.exe --model plamo2-1b-alignment-id1397-dpo-auto.gguf --prompt "hello" --seed 42 --device OPENVINO0 -ngl 999 -c 128 -t 1 -tb 1 *> intermediate-dumps/npu_layers.txt

3) Run CPU dump (CPU-only build):

    $env:LLAMA_EVAL_CALLBACK_TENSOR_PREFIX = "ffn_residual-"
    ./build-cpu/bin/Release/llama-eval-callback.exe --model plamo2-1b-alignment-id1397-dpo-auto.gguf --prompt "hello" --seed 42 --device none -ngl 0 -c 128 -t 1 -tb 1 *> intermediate-dumps/cpu_layers.txt

4) Run diff script:

    ./.venv/Scripts/python.exe scripts/diff_eval_callback_sums.py intermediate-dumps/cpu_layers.txt intermediate-dumps/npu_layers.txt

Expected output: a short text report with the first layer index that differs and per-layer deltas.

## Validation and Acceptance

Acceptance is met when:

- `llama-eval-callback` completes successfully on both CPU and NPU runs for the specified model/prompt/seed without crashing.
- A machine-readable (or at least stable text-parsable) artifact exists for each run.
- The diff report clearly identifies either:
  - the first layer where values differ, or
  - that all compared layers match within a defined tolerance.

## Idempotence and Recovery

- All dump commands write to `intermediate-dumps/` and can be rerun safely after deleting prior outputs.
- If OpenVINO `NPU` crashes, rerun with additional diagnostics enabled (after implementing `try/catch` logging) or set `GGML_OPENVINO_FORCE_DYNAMIC=1` to get a non-crashing run for comparison purposes.

## Artifacts and Notes

- CPU full-node dump: `intermediate-dumps/cpu_eval_callback.txt`
- Initial NPU attempt output (terminated): `intermediate-dumps/npu_eval_callback.txt`

