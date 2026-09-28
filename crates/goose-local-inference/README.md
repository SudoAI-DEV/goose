# goose-local-inference

On-device model inference for goose. Runs GGUF models through `llama.cpp` (via
`llama-cpp-2`), with an optional eredu backend on Apple silicon for SafeTensors and GGUF.

Reach it through [`goose-providers`](../goose-providers) with the
`local-inference` feature, which exposes `LocalInferenceProvider` as an ordinary
`Provider`.

## Features

Default is `[]` — CPU inference.

- `hf-hub` — Hugging Face model discovery, downloads, cache inventory, and
  management APIs. Without it, models can still be loaded directly from paths.
- `cuda`, `vulkan` — GPU acceleration via the corresponding `llama-cpp-2` backend.
- `mlx` — eredu with its MLX/Metal execution backend on Apple silicon.
  The feature name is retained for compatibility.

## What it handles

- **Runtime and placement** — `InferenceRuntime` describes the machine and
  `available_inference_memory_bytes` helps choose a cached model that will fit.
- **Model lifecycle** — `is_model_loaded`, `loaded_model_ids`, and `evict_model`
  manage what's resident. With `hf-hub` enabled, `hf_models` uses the Hugging
  Face cache as the model inventory, `management` exposes it to clients, and
  `huggingface_auth` handles gated repos.
- **Prompt formatting** — `prompt_template` applies the model's chat template;
  `builtin_chat_template_names()` lists the bundled ones.
- **Tool calling** — eredu's semantic events and llama.cpp's `tool_parsing`
  extract tool calls from model output, and `tool_emulation` (toolshim) fills in
  for models with no native tool support.
- **Richer outputs** — `thinking_output` separates reasoning blocks from the
  answer; `multimodal` handles image input.
- **Config** — `config_resolver` and `provider_utils` resolve settings such as
  `LOCAL_LLM_MODEL`.

## Loading models from a path

Set the model name to a local path to bypass the Hugging Face cache. A `.gguf`
file defaults to llama.cpp. A SafeTensors model directory containing
`config.json`, a supported tokenizer (`tokenizer.json`, `tokenizer.model`, or
`spiece.model`), and weights defaults to eredu; a path to one of its
`.safetensors` files is accepted as well. Relative paths
are resolved from the process working directory.

Models loaded this way remain user-owned: Goose can load and evict them from
memory, but does not include them in the cached-model inventory or delete their
files.

## Selecting an inference backend

The desktop model settings expose **Inference backend**. Automatic selects
llama.cpp for GGUF and eredu for SafeTensors. Choose eredu explicitly to run a
GGUF with MLX. This reuses the cached checkpoint and model ID; switching engines
does not download another copy. Available choices reflect the platform and build;
eredu validates the particular checkpoint when loading it.

Programmatic callers can set `ModelSettings.backend_id` to `eredu` or `llamacpp`,
or leave it unset for automatic selection. Existing `mlx` settings are accepted
as an alias for eredu. Goose persists these settings under
`GOOSE_LOCAL_MODEL_SETTINGS`, keyed by model ID. llama.cpp cannot load SafeTensors.
An explicitly selected backend never silently falls back to another engine.

Download requests may supply `format` (`gguf` or `safetensors`) independently of
`backendId`. Old `mlx` requests and `mlx-safetensors` format values remain accepted.

Eredu uses its prepared-chat pipeline for native tools, reasoning, sampling,
cancellation, and external drafting (including supported Gemma assistants). Auto
tool mode uses Goose's tool emulation when the checkpoint/template cannot support
native tools. Force-native mode reports the incompatibility. Tool calls are emitted only after completion;
partial argument streams never become executable Goose tool requests.

### Eredu compatibility

- The dependency is pinned to git main revision
  `ee7670e11bc0704b340e1fa4e24bd5780f8fe687` across the eredu crates.
- Checkpoint architecture, tensor encoding, tokenizer, and processor must be
  supported by eredu; GGUF compatibility with llama.cpp does not imply eredu
  compatibility.
- Embedded and custom Jinja templates are supported. The named built-in templates
  exposed by llama.cpp are specific to that engine; use embedded metadata or the
  full Jinja source with eredu.
- GGUF projectors must be discoverable by eredu beside the checkpoint (or in its
  supported parent search locations). Ambiguous companions or a selection that
  differs from Goose's projector are rejected. Select llama.cpp for layouts
  requiring an explicit projector override.
- Image attachments are decoded locally. Prompt usage and context budgeting use
  eredu's count of actual model positions in the prepared input, including media;
  `image_token_estimate` is not used by this backend.
- Draft models must be supported eredu target–assistant pairings. Changing the
  draft model reloads the eredu session. Mirostat works with speculation, but
  optimistic lookahead is disabled in this adapter. Goose currently selects
  external drafting only; it has no setting for eredu's embedded prediction heads.
- The Apple backend targets macOS 14+ and requires Xcode 26.2+ for its native
  build. Other platforms retain llama.cpp; eredu is not enabled there by this
  integration.

### Memory planning and remaining gaps

Goose includes `memory_forecast` in its existing eredu inference response log.
The forecast borrows the exact prepared request before generation, using eredu's
ordinary or speculative API as appropriate. It retains the full estimate,
assumptions, execution contract, and speculative plan when available. Forecast
errors are recorded as `error` and do not reject generation. An already-cancelled
request has no forecast. These are loaded-model estimates: they exclude loading
peaks and are recorded with the response, not exposed as a preflight UI check.

Eredu now automatically caps an untouched native MLX allocator-cache default at
256 MiB when realizing a model, preserving smaller defaults and explicit native
settings. Goose inherits that process-global policy. This limits retained reusable
allocator blocks, not active tensors, KV state, or total process memory; it can
trade some throughput for lower retention. Goose does not expose a cache-policy
override or change the limit on each request.

Forecasts are descriptive estimates, not allocation limits or reservations.
Goose does not yet use them for admission, model recommendations, or automatic
prefill tuning. Cached-model recommendations still compare checkpoint file size
with available memory, without request-specific KV state, workspace, or loading
peaks. Forecasts use eredu's default calibration and observed availability, with
no application budget or extra reserve configured by Goose.

The existing forecast calls pick up eredu's execution-derived workspace accounting,
including attention tile retention, resident parameter conversions, dense LFM2
hybrid workspaces, and supported routed mechanisms. Forecast fields are serialized
in full, preserving new coverage and missing-mechanism details. These improvements
do not imply complete coverage for every checkpoint in a model family. Reducing
a requested chunk size cannot reduce memory for a path requiring full-pass
prefill, such as the current LFM2 path.

Upstream bounds independent autoregressive drafting when both loaded models have
workspace coverage. It also supports embedded startup and settled continuation
forecasts for covered mechanisms, including an all-attention dense Qwen3.5 text
configuration. Released Qwen hybrid schedules still have gaps for gated-delta
recurrence and some expert reductions. Feature-conditioned external assistants
(including Gemma assistants), prepared media, transfers, and other uncovered
mechanisms can still have unknown upper bounds. Goose cannot currently select
the embedded mode, so that new upstream coverage is not exposed here.

Eredu supports forecasts at settled ordinary, independent speculative, and
covered embedded speculative boundaries. Goose's callback-based generation does
not expose controlled sessions or sample continuation forecasts. It resets
request state and replays the conversation each turn, so KV prefix reuse remains
unimplemented. Arbitrary in-flight outlooks and feature-conditioned external
assistant continuations remain upstream gaps.

Optional retained F32 weight conversions now have their own managed 256 MiB
budget, separate from the allocator cache. Goose leaves the execution-plan
override unset and inherits that finite default. The bound covers retained
payload plus outstanding reservations across the target's eligible owners;
embedded prediction shares it, while each separately loaded external drafter has
its own budget. Unsupported retention paths report an effective disabled policy.
Unlimited retention requires an explicit upstream override that Goose does not
request. This resolves the earlier unbounded-retention gap.

Eligible mixed-dtype Metal projections now read the original narrow weights
without full-weight F32 casts during decode and supported multi-row prefill or
speculative verification. The multi-row path requires materialized row-contiguous
F16/BF16 weights, F32 activations, supported geometry (currently 2–2,000 rows), and
native SIMD GEMM dispatch. NAX/TF32 and other unsupported cases retain the native
fallback and may still need temporary casts. Goose inherits the selector without
changing native arithmetic settings. Neither retention nor allocator-cache policy
limits those temporaries, native backing capacity, KV state, or total process memory. Request
reset preserves admitted conversions; dropping their owner releases its claims.
Eredu also exposes settled trimming, but Goose does not currently expose trimming
or custom conversion-retention budgets.

Goose's existing loaded forecasts include the new retention policy, scope, usage,
and reservation accounting without adding retained payload to resident parameters
twice or clamping temporary workspace to the retention budget. These observations
precede generation, so use a subsequent request's forecast to inspect warmed
residency; Goose does not yet expose a separate post-generation retention report.

Loaded forecasts now use proven binding, dtype-flow and native dispatch facts to
remove unnecessary promotion allowances, while retaining split-K partials and
possible input copies. Cold forecasts remain conservative where those facts are
not available. Coverage for a short request does not imply coverage for a longer
prefill or a different architecture.

See the pinned [upstream memory documentation](https://github.com/jbg/eredu/blob/ee7670e11bc0704b340e1fa4e24bd5780f8fe687/doc/generation-memory.md)
and [mixed-storage projection contract](https://github.com/jbg/eredu/blob/ee7670e11bc0704b340e1fa4e24bd5780f8fe687/doc/mixed-storage-gemm.md)
for coverage, calibration assumptions, and validation scope.

### Recommendations before downloading weights

The local inference settings page discovers candidates when its search box is empty.
Candidates come from HF trending GGUF/MLX repositories and quantizations of models
on HF benchmark leaderboards; Goose maintains no model catalogue. The quality
policy currently uses MMLU-Pro, IFStruct v1.0 and Terminal-Bench 2.0 as separate
ranking cohorts, in that order. Scores from unrelated benchmarks are not averaged.
Reported and verified scores retain their provenance and setup notes. Only an
explicit or HF-tagged, unambiguous quantization relationship inherits an upstream
checkpoint's scores; fine-tunes, adapters and merges do not. Inherited scores are
labelled and are not measurements of the quantized variant.

Discovery returns lightweight results first. The desktop progressively requests
variant listings and then one assessment per variant, with two background repository
workers and an initial twelve-candidate inspection page. “Check more models” extends that budget.
Typing a new query stops scheduling work for the previous results and ignores its
late responses. Already dispatched ACP requests may finish in the background.
Search order is retained; the settings landing page orders assessed candidates by
fit, benchmark cohort/rank and popularity, and deduplicates scored checkpoints.
At most three scored candidates receive a recommendation; fitting unscored models
are labelled separately. Unknown or failed assessments remain visible.

With `mlx` compiled on Apple silicon, `recommendations` pins the HF manifest's
commit SHA, supplies complete SafeTensors headers or bounded GGUF prefixes to
`inspect_model_metadata`, and calls `forecast_inspected_generation`. Configuration,
indexes, all shards and applicable processor/projector metadata use that same
revision. Range responses must be HTTP 206 with a matching Content-Range; ignored
ranges are rejected before consuming the response. Individual metadata objects
are limited to 16 MiB, aggregate shard headers to 64 MiB and shard sets to 256.
Oversized/incomplete metadata produces an unknown assessment, not a memory rejection.
GGUF prefix reads can extend into payload bytes within the prefix limit; only the parsed header
is supplied to eredu. No complete checkpoint download is initiated by discovery.

The profile is text-only, 4,096 input and 1,024 output tokens, batch one, fully
resident weights, no draft model, eredu's default conversion-retention policy and
forecast calibration, a 256 MiB allocator-cache allowance and a 1 GiB reserve.
It includes loading peaks. Hardware availability is sampled for each request;
profile assumptions, bounds, uncertainty and revision are returned to the UI.
An eredu fit is accepted as a llama.cpp fit estimate. An eredu shortfall or
incompatibility does not establish that llama.cpp cannot run the GGUF, and does not
disable its download. SafeTensors still requires a build with eredu support.

HF JSON is cached in memory for ten minutes, partitioned by credentials and capped
at 32 MiB. Up to 128 immutable forecasts are cached separately; current capacity is
reapplied without fetching headers again. This is also the conservative prefilter:
previously inspected revisions need no new tensor metadata even when their known
lower bound exceeds current capacity. For a first encounter, repository size,
parameter count and quantization names do not establish an unavoidable resident
memory lower bound. They are never used to reject a candidate before inspection.

Remaining limitations:

- Benchmark coverage, evaluation settings and quantization lineage are incomplete.
  HF-reported scores are not necessarily independently verified. Popularity is a
  separate fallback, never a fabricated quality score. The quality module isolates
  the HF source so another benchmark feed can replace or supplement it.
- Discovery is bounded and is not an exhaustive search for the globally best model.
  The first page may contain pending, unknown or oversized candidates while fit
  estimates arrive. Additional candidates remain available through search/pages.
- Cold forecasts preserve eredu's unbounded costs. They cannot establish tokenizer,
  chat-template, native-tool readiness, payload integrity or actual performance.
  Custom contexts, image input, speculation and non-default cache policies need a
  new forecast. The recommendation profile does not change saved model settings.
- Downloads resolve the repository again; loading still inspects the actual local
  artifact. A recommendation at one revision is not admission for a newer revision.
- There is no persistent metadata cache, configurable benchmark policy, automatic
  smaller-context alternative, or post-download memory preflight yet.

See the pinned [metadata forecasting contract](https://github.com/jbg/eredu/blob/ee7670e11bc0704b340e1fa4e24bd5780f8fe687/doc/metadata-forecasting.md)
and [HF leaderboard API](https://huggingface.co/docs/hub/leaderboard-data-guide).

### Recommended Goose follow-up work

In priority order:

1. Extend discovery forecasts to the user's actual context, media and draft-model
   settings; add post-download/load preflight and configurable reserves. Compare
   smaller-context alternatives while preserving eredu's unknown bounds.
2. Expose per-model conversion-retention policy, live target/drafter usage and a
   settled trim action. Use the upstream observations and trimming APIs rather
   than a second accounting system. Policy changes require reload; trimming does
   not promise an equivalent drop in process memory.
3. Add an eredu prefill-chunk setting and compare alternatives with
   `GenerationForecast::with_prefill_chunk`. Apply it only where the selected
   execution contract supports chunking; smaller chunks cannot help full-pass
   paths merely because the UI accepts the setting.
4. Expose embedded drafting separately from an external draft model, with supported
   proposal-count and lookahead controls. Include the complete drafting plan in
   the model-cache identity so changing settings reloads the session correctly.
5. Adopt controlled sessions for pause/resume, settled continuation forecasts and
   trimming during a paused request. Investigate KV-prefix reuse across agent turns
   separately: cached state must match the exact rendered conversation, including
   tool results and template changes. Controlled execution alone does not provide
   that matching policy.

The new `projection-profiling` feature is useful for a separate developer diagnostic
build when investigating performance. It retains tensor samples and changes memory
and timing, so ordinary Goose builds should keep it disabled. The projection
optimizations themselves need no new Goose feature flag.

## Building

The `llama.cpp` backends compile native code, so a C/C++ toolchain is required,
plus the CUDA or Vulkan SDK when selecting those features.

```bash
cargo build -p goose-local-inference
cargo build -p goose-local-inference --features hf-hub
cargo build -p goose-local-inference --features mlx
```
