# Fix OpenVINO CPU crash at first decode eval for `plamo-2-translate`

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.
This repository includes `.agents/PLANS.md` at the repo root; this plan must be maintained in accordance with that file.

## Purpose / Big Picture

Enable `llama-cli.exe` on Windows to generate output tokens (at least 8 tokens with `-n 8`) when using the OpenVINO backend (device `OPENVINO0`) with `GGML_OPENVINO_DEVICE=CPU` for the model `plamo-2-translate_Q4_0.gguf`.

Before this work, prompt prefill succeeded, but the process crashed (Windows `0xC0000409`, stack buffer overrun) at the moment it starts evaluating the first generated token after prefill (the “first decode eval”), inside OpenVINO `InferRequest::set_input_tensor(...)`. After this work, the same command prints tokens and exits with code 0.

## Progress

- [x] (2025-12-24T18:28:05+09:00) Added `GGML_OPENVINO_TRACE` logging + `FrontEnd::convert` exception reporting to make OpenVINO failures observable instead of aborting silently.
- [x] (2025-12-24T18:28:05+09:00) Fixed OpenVINO frontend translation issues encountered during bring-up (VIEW/CONT/MatMul shape handling, GQA head broadcast).
- [x] (2025-12-24T20:30:00+09:00) Added CPU fallbacks for OpenVINO-CPU-unstable ops (notably `SSM_SCAN`, `SSM_CONV`, plus a few additional ops) to avoid `compile_model` crashes.
- [x] (2025-12-24T21:05:00+09:00) Reproduced the remaining crash reliably with `GGML_OPENVINO_TRACE=1`: the crash happens during `set_inputs` for a cached graph beginning with `mamba_conv1d_silu-*`.
- [x] (2025-12-24T21:15:00+09:00) Fixed the OpenVINO dynamic-graph cache key so prefill and decode graphs with different input shapes do not share the same cached `InferRequest`.
- [x] (2025-12-24T21:20:00+09:00) Verified the user reproduction command (`-n 8`, `-lv 3`, `GGML_OPENVINO_DEBUG_OUTPUT=1`) prints output and exits successfully (`EXITCODE=0`).

## Surprises & Discoveries

- Observation: `-n 1` may appear to “work” even when decode-eval is broken, because `llama-cli` can sample the 1st generated token from the prompt logits without evaluating that token (evaluation is only needed to produce logits for the next token).
  Evidence: the crash reproduces with `-n 8` but not necessarily with `-n 1`.

- Observation: The remaining failure mode was not an OpenVINO exception but a hard process crash during `InferRequest::set_input_tensor` (Windows `0xC0000409`).
  Evidence: `GGML_OPENVINO_TRACE=1` consistently ended at `set_input[0] 'mamba_conv1d-0'` followed immediately by process termination.

- Observation: The crash was caused by reusing a cached OpenVINO compiled graph across different input shapes between prefill and decode (token-length dimension changes), due to an overly coarse cache key.
  Evidence: the cache key previously used only `(n_nodes, first_node_name, last_node_name)` and did not include any input-shape signature, so a prefill-compiled graph could be reused during decode even when an external input (e.g. `mamba_conv1d-0` produced by a CPU fallback op) changed shape.

## Decision Log

- Decision: Make the OpenVINO graph cache key shape-aware by hashing external input tensor signatures (name + type + shape + stride) in addition to `(n_nodes, first_node_name, last_node_name)`.
  Rationale: OpenVINO CPU may hard-crash when `InferRequest::set_input_tensor` is called with an input tensor shape incompatible with the compiled model’s parameters. A cache key that includes input signatures prevents accidental reuse across shape changes (prefill vs decode), eliminating the crash without needing dynamic shapes for every intermediate tensor.
  Date/Author: 2025-12-24 / Codex

## Outcomes & Retrospective

Outcome: The command below now produces output and exits with code 0 on Windows with OpenVINO CPU.

Key lesson: OpenVINO graph caching must include shape information for any graph whose external inputs can vary by token-length or phase (prefill vs decode). Keying only on graph size and a couple of node names is not safe.

## Context and Orientation

Repro environment:

- OS: Windows + PowerShell
- Repo root: `C:\Users\mitmul\llama.cpp`
- Model: `plamo-2-translate_Q4_0.gguf` (repo root in this workspace)
- Binary: `build-openvino-cpu-relwithdebinfo\bin\RelWithDebInfo\llama-cli.exe`
- OpenVINO: must be installed and its DLLs must be on `PATH` (typically by running OpenVINO’s `setupvars.ps1`).

How this fails (plain language):

- `llama-cli` evaluates the prompt (“prefill”) and then starts generating tokens.
- When `-n` is greater than 1, `llama-cli` must evaluate the first generated token to obtain logits for the second token.
- During that first decode evaluation, ggml schedules some ops on CPU (because OpenVINO CPU is unstable for them), and later ops on OpenVINO. That means some OpenVINO graphs take external inputs that come from CPU ops, and those shapes differ between prefill and decode.
- The OpenVINO backend caches compiled graphs (via `ov::InferRequest`). If the cache reuses a prefill-compiled graph for a decode step with different external input shapes, OpenVINO may crash during `set_input_tensor`.

Relevant files (paths are repo-root relative):

- `ggml/src/ggml-openvino/utils.h`: `graph_key` used to cache compiled models / infer requests.
- `ggml/src/ggml-openvino/utils.cpp`: `ov_graph_compute_dynamic(...)` caching logic + `compute_graph_key(...)` implementation.
- `ggml/src/ggml-openvino/ggml-openvino.cpp`: CPU fallbacks for unstable ops on OpenVINO CPU.

## Plan of Work

1. Reproduce the crash at the first decode eval with `GGML_OPENVINO_TRACE=1` and confirm the last trace line is inside `set_inputs`.

2. Confirm whether the crashing graph is a cache hit, implying an earlier compilation (often from prefill) is being reused.

3. Fix caching so graphs that differ in external input shapes cannot share the same cache entry:

   - Extend `graph_key` to include an `inputs_hash` field.
   - Implement `compute_graph_key(...)` to hash the signatures of external input tensors (tensors used as `src[...]` but not produced by nodes in the current `ggml_cgraph`).

4. Rebuild and re-run the reproduction command until `-n 8` completes and prints output.

## Concrete Steps

All commands run from `C:\Users\mitmul\llama.cpp`.

1. Enable OpenVINO runtime DLLs (this path matches `build-openvino-cpu.ps1` defaults in this repo; adjust if yours differs):

    & "C:\\Users\\NECPC-USER\\Downloads\\openvino_genai_windows_2025.4.0.0_x86_64\\openvino_genai_windows_2025.4.0.0_x86_64\\setupvars.ps1" | Out-Host

2. Rebuild:

    cmake --build build-openvino-cpu-relwithdebinfo --config RelWithDebInfo

3. Run the user reproduction:

    $Env:GGML_OPENVINO_DEVICE="CPU"
    $Env:GGML_OPENVINO_DEBUG_OUTPUT=1
    .\\build-openvino-cpu-relwithdebinfo\\bin\\RelWithDebInfo\\llama-cli.exe `
        -m .\\plamo-2-translate_Q4_0.gguf `
        -c 2048 `
        -n 8 `
        --no-warmup `
        -lv 3 `
        -p "<|plamo:bos|><|plamo:op|>dataset\ntranslation\n<|plamo:op|>input\nThis is a white pen.\n<|plamo:op|>output\n" `
        -sp `
        --single-turn

   Expected: output tokens are printed and the process exits with code 0. The log should contain 8 occurrences of `n_remain:` counting down to 0.

## Validation and Acceptance

Acceptance is satisfied when:

- The reproduction command prints output and exits successfully (Windows exit code 0).
- The run does not terminate with `0xC0000409` during the first decode evaluation.
- With `GGML_OPENVINO_TRACE=1`, the run progresses past `set_input[0] 'mamba_conv1d-0'` (or similar) without crashing.

## Idempotence and Recovery

- Rebuilding with `cmake --build ...` is safe to repeat.
- If you need to compare before/after behavior, `GGML_OPENVINO_TRACE=1` is the fastest way to identify where the process terminates.

## Artifacts and Notes

- Debugging tip: redirect stdout/stderr to a log file, because `GGML_OPENVINO_DEBUG_OUTPUT=1` can be verbose:

    .\\build-openvino-cpu-relwithdebinfo\\bin\\RelWithDebInfo\\llama-cli.exe ... *>&1 | Out-File repro.txt -Encoding utf8

- After this fix, `repro_user_lv3_after_cachekeyfix.txt` in the repo root is an example captured run showing `n_remain:` reaching 0 and `EXITCODE=0`.

## Interfaces and Dependencies

- OpenVINO compiled graph caching:
  - `ov_graph_compute_dynamic(...)` stores `std::shared_ptr<ov::InferRequest>` in an `unordered_map<graph_key, ...>`.
  - `graph_key` must uniquely identify cases where the OpenVINO model expects different input shapes, otherwise `set_input_tensor(...)` can crash on OpenVINO CPU.

---

Revision note (2025-12-24 / Codex): Rewrote this ExecPlan to reflect the actual remaining crash (OpenVINO `set_input_tensor` during first decode eval) and the implemented fix (shape-aware cache key), and marked validation as completed with a captured successful run.
