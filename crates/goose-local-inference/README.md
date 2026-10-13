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
Goose does not yet use runtime forecasts for admission or automatic prefill tuning.
Remote model recommendations use a separate metadata-only forecast profile with
an explicit reserve, described below. Cached models are not recommended using
checkpoint file size alone.

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

The local inference settings page uses the checked-in
[`data/model-catalogue.json`](data/model-catalogue.json). On an eredu-enabled build,
the empty-search landing page recomputes memory fit entirely offline, ranks by
benchmark evidence, deduplicates evaluated checkpoints and shows up to ten eligible
recommendations. Catalogue membership also supplies recommendations in explicit HF
search results. No benchmark or model metadata is downloaded for the landing page,
including models that cannot fit the current machine.

Rebuild on Apple silicon with eredu support:

```sh
just rebuild-local-model-catalogue
just check-local-model-catalogue
```

The rebuild downloads metadata, never full checkpoints, and does not open a Metal
device. Each eredu inspection runs in a fresh subprocess, with at most two active
inspection workers, so inspection allocations cannot accumulate over the sweep.
The parent fetches metadata for up to eight checkpoint families and owns shared HF
rate-limit handling. Worker failures abort publication. A full refresh can take hours when many sharded conversions need inspection;
the command uses an optimized build and the release job allows six hours. The
offline check also works without eredu or Apple hardware. For a single
repository diagnostic, run the `local-model-catalogue` binary with
`--explain owner/model`. The rebuild writes rejection reasons and source errors to
`target/local-model-catalogue-report.json`; completed checkpoint families are cached
beside that report for up to 24 hours to resume interrupted refreshes. Delete the
`.cache` and `.metadata` directories to force a completely fresh refresh. Public
metadata is cached there for the same 24-hour window; HF rate-limit windows are
honored with shared backoff across workers. The existing catalogue is replaced
atomically only after successful generation and validation. Transient source errors
fail the refresh rather than publish a depleted list.

Release preparation rebuilds on a macOS runner, downloads the resulting artifact
into the version-bump or patch-release PR, and validates it before committing. Release
packaging consumes that committed file. When preparing a release manually, rebuild
before `just prepare-release`. Schema/policy, forecast-profile or eredu-pin changes
invalidate the old catalogue and require regeneration.

The generator uses the Artificial Analysis Intelligence Index as its sole ranking
source. It fetches scored models through AA's authenticated API and resolves exact
HF checkpoint IDs from the weights links on AA model pages, without fuzzy name
matching. Ambiguous weights links fail the refresh. Models without HF weights links
are omitted. For multiple evaluated configurations of one checkpoint, the highest
measured index score wins; the evaluation name and AA model link identify that configuration.
The snapshot records the published methodology version and generation date.
API scores are used only in memory to order checkpoints. The distributed catalogue
contains the ordering and provenance, not scores or numeric ranks. Runtime filters
preserve that order; no AA requests or credentials are needed in the desktop or CLI.
Models with no AA evaluation are searchable and downloadable but not recommended.
AA's API omits estimate status. The builder reads the explicit
`intelligenceIndexIsEstimated` flag from each weights-linked model page, matching
its AA model ID. Estimated configurations are excluded before choosing the best
configuration for a checkpoint. Missing, conflicting or malformed status fails the
refresh and preserves the existing catalogue. A change in AA's page structure may
require a parser update; unknown status is never treated as measured.
Evaluation reasoning settings and quantization can differ from the local configuration.

Set `ARTIFICIAL_ANALYSIS_API_KEY_FILE` to a file containing only your API key, or
set `ARTIFICIAL_ANALYSIS_API_KEY` in the release environment. Keys must not be
committed or embedded in applications. The builder does not log or cache keys.
The UI attributes Artificial Analysis and links to the evaluated configuration.
Redistribution permission for the ordered catalogue remains to be confirmed with AA;
this integration is currently for local evaluation before publication.

Only unambiguous quantizations inherit scores;
fine-tunes, adapters, distillations, abliterations and merges need their own evaluations.
New modification claims in repository names, tags or card model names also prevent
inheritance. Conflicting relation tags or datasets
absent from the evaluated checkpoint make inheritance ambiguous and exclude that
conversion. This can also exclude calibration-only conversions whose metadata does
not distinguish calibration from training. Scores inherited from a full-precision
checkpoint are not measurements of the conversion.

Each inherited variant must also match the evaluated checkpoint's architecture and
structural dimensions (hidden size, layers, attention/KV heads, vocabulary, and
feed-forward width/expert count where declared). The builder reads revision-pinned
config.json or bounded GGUF header ranges, records both identities and the evaluated
revision, and rejects missing or conflicting evidence. Known GGUF architecture aliases
are format mappings, not model allowlists. No weights are downloaded for these checks.
A rejection appears in the builder report or `--explain` output.

To reapply lineage policy to existing, immutable catalogue entries without repeating
Eredu forecasts, run `local-model-catalogue --revalidate-lineage [catalogue.json] [report.json]`
with the same features as the rebuild command. This only removes ineligible entries;
run a full rebuild to discover replacements. It requires the same Eredu revision and
forecast profile; source failures leave the catalogue unchanged.

For each evaluated checkpoint and conversion class (format and quantization), the
generator elects the first eligible artifact, considering the original checkpoint
then publishers in download-count order. At most three artifacts per class undergo
inspection. It imposes no model-size limit and never consults the generator host's
RAM. Duplicate publishers are not all stored. Every accepted variant contains an
immutable revision, exact download manifest, suitability evidence and eredu's
computed lower/upper memory bounds and uncertainty, without host memory budgets.
Runtime compares those bounds plus reserve with current capacity. During generation,
these comparisons are checked against eredu at the fit boundaries. Native GGUF block
metadata cannot be serialized in a `GenerationMemoryRequest`, so the catalogue stores
the computed bounds rather than execution plans.

HF task, purpose tags, architecture heads and chat configuration provide suitability
evidence; model names and hand-maintained model IDs are not used. Non-conversational
tasks, reward/embedding/classification models and explicitly tagged base models are
excluded. Task alone or a knowledge benchmark alone does not establish conversational
use. Instruction/agent evaluations, conversational tags or declared chat configuration
provide positive evidence. Missing metadata stays unknown. Only unambiguous
quantizations inherit suitability from an upstream checkpoint.

Suitability, eredu compatibility and memory fit remain independent assessment fields.
Recommendations require positive suitability and compatibility, plus supported
benchmark evidence. Memory eligibility includes likely fits and forecasts with no
upper bound whose measured lower bound plus reserve fits the available budget.
Missing lower bounds, unavailable capacity and bounded forecasts that straddle the
budget do not qualify. Unbounded recommendations retain the “Memory fit unknown”
label; a likely-fitting variant is preferred within a checkpoint when available.
Explicit search retains its result order and candidates. Models outside the catalogue
are not recommended, but absence alone does not disrecommend them. Their variants
can still receive live assessments: twelve repositories are inspected initially and
“Check more models” extends that budget. Shortfalls, unsuitable models and incompatible
GGUFs remain downloadable with explanations when their backend is available.
Recommended downloads use the assessed immutable revision; a per-variant HF cache
reference ensures later loading selects that snapshot rather than a stale `main`.
A different cached revision is not marked downloaded for a catalogue entry. A
successful download evicts any loaded execution after its active request finishes,
so the next load uses the selected snapshot.

With `mlx` compiled on Apple silicon, `recommendations` pins the HF manifest's
commit SHA, supplies complete SafeTensors headers or bounded GGUF prefixes to
`inspect_model_metadata`, and calls `forecast_inspected_generation`. Configuration,
indexes, all shards and applicable processor/projector metadata use that same
revision. SafeTensors config identity and the first GGUF shard's architecture are
screened through eredu's own resolvers before fetching remaining tensor metadata.
Full inspection still establishes exact architecture/layout/quantization/backend
compatibility. Range responses must be HTTP 206 with a matching Content-Range; ignored
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
at 32 MiB. Up to 128 immutable forecasts or definitive incompatibilities are cached
separately by repository, revision, format and file. The process-local cache cannot
outlive the compiled eredu version. Transient failures and unverified compatibility
are not cached as rejections. Current capacity is reapplied to cached forecasts
without fetching headers again. This is also the conservative prefilter:
previously inspected revisions need no new tensor metadata even when their known
lower bound exceeds current capacity. For a first encounter, repository size,
parameter count and quantization names do not establish an unavoidable resident
memory lower bound. They are never used to reject a candidate before inspection.

Remaining limitations:

- Benchmark coverage, evaluation settings and quantization lineage are incomplete.
  Estimated AA index values are excluded; HF conversion provenance is not necessarily
  independently verified. Structural checks cannot detect undeclared weight changes that preserve
  architecture and dimensions, nor establish that today's upstream revision is the
  exact revision benchmarked. Conservative missing-evidence checks can omit valid
  conversions, including those whose upstream config is gated from the public builder.
  Popularity is only a tie-breaker within an evaluated checkpoint and a
  representative-publisher signal. The quality module consumes AA as the sole source.
- The catalogue is limited by benchmark coverage and the three-artifact inspection
  budget per conversion class. It can miss suitable models or quantizations. Candidates
  with unknown compatibility/suitability, missing memory evidence, oversized lower
  bounds or unsupported execution remain available through explicit search but
  do not fill the recommendation section. Catalogue repositories show the stored
  representative variants; newly published variants appear after a refresh.
- Cold forecasts preserve eredu's unbounded costs. They cannot establish tokenizer,
  chat-template, native-tool readiness, payload integrity or actual performance.
  Declared template metadata is only suitability evidence. eredu's local
  `inspect_text_model` skips supplied remote metadata; validating text sidecars
  before download requires a corresponding upstream metadata inspection API.
  Custom contexts, image input, speculation and non-default cache policies need a
  new forecast. The recommendation profile does not change saved model settings.
- Compatibility describes the pinned eredu build. Some otherwise common GGUFs are
  rejected for tensor-binding differences; for example, the inspected GPT-OSS GGUF
  uses `post_attention_norm` where eredu expects `attn_post_norm`. Some Gemma 2
  conversions also lack the normalization tensor names eredu expects. Some Qwen 3.5/3.6
  shared-expert gate layouts conflict with eredu's expected ranks. Some DeepSeek
  conversions lack required MLA metadata or use incompatible metadata types.
  Some Llama 3 GGUFs contain an unadmitted `rope_freqs.weight` tensor.
  eredu also applies its default 2 GiB
  per-tensor GGUF allocation limit during both remote inspection and local artifact
  loading; its facade does not expose an override, so some large expert tensors
  remain ineligible regardless of host RAM. Rejections are recorded in the refresh
  report. Loading still inspects the actual local artifact.
- There is no configurable benchmark policy, automatic smaller-context alternative
  or post-download memory preflight yet. A release catalogue ages until refreshed.

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
