# qwen36-mtp

High-performance local serving of **Qwen3.6 with Multi-Token Prediction (MTP)**
on a single 24 GB consumer GPU (tested on RTX 4090; should also work on RTX 3090
with `CUDA_ARCH=86`).

The point of this repo is to capture the *exact* recipe — upstream
[llama.cpp](https://github.com/ggml-org/llama.cpp) with merged MTP support,
MTP-preserving GGUF, CUDA-toolkit-via-conda, shared-cudart link fix, sane launch
flags — so that a fresh box can go from zero to a working OpenAI-compatible
endpoint with one command.

## Why this stack?

| Need | What this repo gives you |
| --- | --- |
| Real Qwen3.6 quality on 24 GB | dense 27B (or 35B-class with the right GGUF), q4 weights, q4 KV cache |
| 100+ tok/s decode | NextN/MTP speculative decoding (~3× over no-MTP) |
| Large context with safer 24 GB headroom | 131K default, configurable to ≥196K thanks to hybrid GatedDeltaNet + GatedAttention layout |
| Strong prefix caching | upstream `--cache-idle-slots`, `--ctx-checkpoints`, KV-slot save/restore |
| OpenAI-compatible | `llama-server` with `--jinja` and `--reasoning-format deepseek` |
| Tool-calling | `--jinja` enables Qwen's tool-call template; LiteLLM in front recommended for heavy agents |

## Measured numbers (RTX 4090, selected profiles)

The default is now the quality-first/headroom profile: `CTX_SIZE=131072` and
`PARALLEL=2`. The older 196K/262K profiles remain supported when you need more
context and can spare VRAM.

| ctx | parallel | decode | prefill | MTP accept | VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 768 | 4 | 124 tok/s | 310 tok/s | 76.8 % | 18.8 GB |
| 131 072 | 2 | 47 tok/s | 1 741 tok/s | 45.3 % | 20.7 GB |
| 196 608 | 4 | 142 tok/s | 196 tok/s | 4/4 | 22.2 GB |
| 262 144 | 4 | 159 tok/s | 342 tok/s | 4/4 | 23.8 GB |

131K row: `/completion`, `cache_prompt=false`, 120 071-token cold prompt,
512-token decode, VRAM sampled immediately after the request.

(HF model card reports 100.3 tok/s short-ctx on a 3090 Ti with the same recipe.)

### Prefill-cache TTFT (RTX 4090, 196 608 ctx, 3 628-token prompt)

| call | wall | `prompt_n` re-prefilled | `cache_n` reused via KV-shift |
| --- | ---: | ---: | ---: |
| cold | **3.11 s** | 3628 | 0 |
| warm (same preamble, different tail) | **0.42 s** | 515 | 3112 |
| warm again | **0.38 s** | 514 | 3112 |

~7.5× faster TTFT on subsequent calls thanks to `--cache-reuse 256`,
`--cache-idle-slots`, and upstream llama.cpp's KV-slot persistence.

### Tuning `--cache-reuse` (RTX 4090)

Swept 64 / 128 / 256 / 512 on this stack. **Differences are within noise** for
both pure prefix-match prompts (warm TTFT 0.43-0.45 s across all values) and
prompts with a varying middle (mid-prompt KV-shift fires roughly the same way).
The default of `256` is fine. If your workload has many small differing
fragments and you want to squeeze a few percent more reuse, try 64; otherwise
don't bother.

The bigger wins were already on by default in this repo:

| flag | effect |
| --- | --- |
| `--cache-idle-slots` | keep slot KV warm across requests within a server lifetime |
| `--slot-save-path` | auto-save/restore slot KV to disk; survives `systemctl restart qwen36` |
| `--ctx-checkpoints 8 --checkpoint-min-step 2048` | context snapshots for recurrent/hybrid KV reuse without overly dense checkpoints |
| `--cache-ram -1 --kv-unified` | unlimited RAM-side overflow + the unified-KV mode required by idle-slot caching |

## Install

Prereqs: NVIDIA driver (CUDA 12-capable), `git`, `cmake`, `conda` (Miniconda or
Mamba). Nothing else. The installer fetches its own CUDA toolkit into a conda
env, so a "driver-only" Linux box works.

```bash
git clone https://github.com/Maverobot/qwen36-mtp.git
cd qwen36-mtp
./scripts/install.sh
```

If you previously installed the old `crucible-mtp` fork, stop the service and
rerun the installer. It fetches `ggml-org/llama.cpp` into the existing checkout,
leaves old fork branches intact, and builds detached upstream `master`:

```bash
systemctl --user disable --now qwen36-multi 2>/dev/null || true
rm -f ~/.config/systemd/user/qwen36-multi.service
systemctl --user stop qwen36 2>/dev/null || true
./scripts/install.sh
systemctl --user daemon-reload
systemctl --user restart qwen36
```

Set `LLAMA_REF=<branch-or-tag>` if you want to pin a specific upstream revision.

Override knobs via env vars. Defaults are quality-first with safer 24 GB
headroom: `CTX_SIZE=131072`, `PARALLEL=2`.

```bash
PREFIX=$HOME/llm/qwen36 \
CTX_SIZE=196608 \
PARALLEL=4 \
PORT=8081 \
./scripts/install.sh
```

If you're on a different GPU, override `CUDA_ARCH` to match it:

| GPU | `CUDA_ARCH` |
| --- | --- |
| RTX 4090 / 4080 / 4070 (Ada)        | `89`  *(default)* |
| RTX 3090 / 3090 Ti / 3080 (Ampere)  | `86` |
| RTX 5090 (Blackwell)                | `120` |

The script creates:

- `$PREFIX/llama.cpp/build/bin/llama-server` — upstream-built binary
- `$PREFIX/models/qwen36-27b-mtp/Qwen3.6-27B-MTP-IQ4_XS.gguf` — ~14 GiB
- `$PREFIX/slot-cache/` — on-disk slot KV cache (auto-save/restore across restarts)
- `~/.config/qwen36-mtp/env` — runtime config (paths, ctx size, port, …)
- `~/.config/systemd/user/qwen36.service` — MTP parallel unit; ExecStart points
  at `scripts/run.sh` *in this repo* so `git pull` updates the launcher.

`qwen36-multi.service` is deprecated and is no longer created by fresh installs.
Old units still work through `scripts/run-multi.sh`, which now shims to the same
MTP + `PARALLEL=4` path and prints a deprecation warning.

## Architecture: single source of truth

```
~/.config/systemd/user/qwen36.service
        │ ExecStart=
        ▼
$REPO/scripts/run.sh                  ← versioned, edit here
        │ reads
        ▼
~/.config/qwen36-mtp/env              ← per-host paths/tunables
        │ pointing at
        ▼
llama-server binary + GGUF
```

Edit the launcher? `git pull` and `systemctl --user restart qwen36`.
Change ctx size or port? Edit `~/.config/qwen36-mtp/env` and restart.

## Run

```bash
~/.config/systemd/user/qwen36.service is installed; just:
systemctl --user daemon-reload
systemctl --user enable --now qwen36
sudo loginctl enable-linger $USER     # keep running across logout

# manual one-shot:
LLAMA_BIN=...llama-server MODEL_PATH=...gguf ./scripts/run.sh
```

Endpoint: `http://localhost:8080/v1` (model alias `qwen3.6-27b`).

```bash
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"reply: pong"}]}'
```

### Parallel MTP is the default

`llama.cpp` merged MTP in `ggml-org/llama.cpp#22673`, and upstream now supports
MTP with parallel slots. This repo therefore uses **one service**:

```bash
--spec-type draft-mtp --spec-draft-n-max 4 --parallel 2 --kv-unified
```

`PARALLEL=2` is written to `~/.config/qwen36-mtp/env` by the installer and is
also the default in `scripts/run.sh`. Tooling (omp, opencode, Copilot CLI) only
needs `http://127.0.0.1:8080/v1`.

Install the unit template once (idempotent):

```bash
ln -sf "$PWD/scripts/systemd/qwen36.service" ~/.config/systemd/user/qwen36.service
systemctl --user daemon-reload
systemctl --user enable --now qwen36
```

Tune slots by editing `PARALLEL` in `~/.config/qwen36-mtp/env` and restarting:

```bash
PARALLEL=1   # lowest memory / old single-slot behavior
PARALLEL=2   # default: light concurrency with safer 24 GB headroom
PARALLEL=4   # higher concurrency for agents/subagents; costs more VRAM
systemctl --user restart qwen36
```

`qwen36-multi.service` is deprecated. If an older install already has that unit,
it will still start `scripts/run-multi.sh`, but the wrapper now unsets stale
`DISABLE_MTP`, forces `PARALLEL=4`, preserves host-specific `PORT` / `ALIAS`, and
prints a deprecation warning. You can remove it after migrating:

```bash
systemctl --user disable --now qwen36-multi 2>/dev/null || true
rm -f ~/.config/systemd/user/qwen36-multi.service
systemctl --user daemon-reload
```

Measured no-MTP multi-slot numbers from the old fork-era profile were useful for
the crossover decision, but upstream parallel MTP replaces that split. Re-run
your local benchmark after migration if exact per-slot throughput matters.

## Use with the GitHub Copilot CLI

The `copilot-wrappers/` folder ships two scripts:

- `copilot-local` — generic. Reads `COPILOT_LOCAL_{BASE_URL,MODEL,API_KEY,MAX_PROMPT_TOKENS,MAX_OUTPUT_TOKENS}` from env, exports them as `COPILOT_PROVIDER_*` / `COPILOT_MODEL`, then exec's `copilot "$@"`.
- `copilot-qwen36-27b` — sets the env for this server and exec's `copilot-local`.

Drop them into a directory on your `PATH` (e.g. `~/.local/bin/` or `~/.dotfiles/.scripts/`) and run:

```bash
copilot-qwen36-27b
```

To wrap another OpenAI-compatible provider, copy `copilot-qwen36-27b` and change
the four env vars at the top.

## Use with omp / opencode

`copilot-wrappers/` also ships wrappers for two other agentic-coding harnesses:

- `omp-qwen36-27b` — for [`@oh-my-pi/pi-coding-agent`](https://www.npmjs.com/package/@oh-my-pi/pi-coding-agent).
- `opencode-qwen36-27b` — for [`opencode`](https://opencode.ai/).

Both harnesses send **multiple `system` messages** per request (a coding system
prompt + an environment block), but the Qwen3.6 chat template embedded in the
GGUF rejects any system message after index 0:

```
Jinja Exception: System message must be at the beginning.
```

Rather than rebuild the GGUF, the wrappers route requests through
`scripts/merge-sys-proxy.py` — a tiny ~120-line streaming HTTP proxy that:

1. Merges all `role=system` messages into a single first system block.
2. Injects `chat_template_kwargs.enable_thinking=false` by default (most coding
   harnesses don't render `reasoning_content`, so they appear to hang while the
   model writes a long internal chain of thought; callers can override by
   sending their own `chat_template_kwargs`).

There is a third subtle bug the proxy also fixes: **HTTP/1.1 framing for the
SSE stream.** `llama-server` returns `Transfer-Encoding: chunked` for streaming
responses; a naive proxy that copies the body but strips/forgets that header
leaves the response with neither `Content-Length` nor `Transfer-Encoding` —
which under HTTP/1.1 means "read until socket close". `curl` happens to be
lenient and just stops at `data: [DONE]`, but opencode's fetch reader follows
the spec and waits forever for the close that never comes (the connection is
keep-alive). The proxy therefore re-emits its own `Transfer-Encoding: chunked`
+ `Connection: close` and chunk-encodes each upstream read as
`<hex-size>\r\n<data>\r\n`, terminated by `0\r\n\r\n`. Without this opencode
hangs even though the server log shows a clean `200 OK`.

Run them with no setup once the server is up:

```bash
omp-qwen36-27b -p "Reply with one word: pong"
opencode-qwen36-27b run "Reply with one word: pong"
```

Each wrapper spawns its proxy on a **distinct port** so you can run several
agents concurrently against the same llama-server on `:8080`:

| Wrapper                       | Client       | Proxy port | Upstream |
| ----------------------------- | ------------ | ---------: | -------- |
| `copilot-qwen36-27b`          | Copilot CLI  |          — | `:8080` direct (Copilot CLI doesn't trip the multi-`system` bug) |
| `copilot-qwen36-35b-a3b`      | Copilot CLI  |          — | `:8080` direct |
| `omp-qwen36-27b`              | omp          |   **8091** | → `:8080` |
| `opencode-qwen36-27b`         | opencode     |   **8092** | → `:8080` |
| `omp-qwen36-35b-a3b`          | omp          |   **8093** | → `:8080` |
| `opencode-qwen36-35b-a3b`     | opencode     |   **8094** | → `:8080` |

So `:8080` is always the model server (one); the `:809x` ports are per-client
adapter shims (one per harness × model). All the proxies forward to the same
upstream — you can keep talking to the bare server on `:8080` (e.g. with
`curl /v1/chat/completions` or the Copilot CLI wrappers) while omp and
opencode each have their own merging-proxy lane.

> **Tip — context window auto-detection.** Both omp and opencode wrappers
> query `$UPSTREAM/props` at launch time and write the running server's
> actual `n_ctx` (e.g. `131072`) into the harness's config: `models.yml` for
> omp, the model's `limit.context` field for opencode. So if you change
> `CTX_SIZE` in `~/.config/qwen36-mtp/env` and restart the server, the
> harness's status-line context indicator follows automatically — no
> manual edit needed. The opencode wrappers also set
> `limit.output` (default 16384, override with `OPENCODE_OUTPUT_LIMIT`).

Per-harness quirks the wrappers paper over:

- **omp** auto-discovers `llama.cpp` providers and hard-codes them to
  `api: openai-responses`, which doesn't match llama.cpp's `/chat/completions`
  endpoint. The wrapper writes `~/.omp/agent/models.yml` with the correct
  `api: openai-completions`, `auth: none`, and (if it exists) busts omp's
  sqlite `model_cache` so a stale `baseUrl` from a previous run can't override
  the new one.
- **opencode** persists a lot of state (`~/.local/share/opencode/`,
  `opencode.db`, snapshots, sessions). The wrapper sets
  `OPENCODE_CONFIG_DIR=~/.config/qwen36-mtp/opencode` so it does not touch
  your normal `~/.config/opencode/opencode.json` or merge with global plugins.
  The `@ai-sdk/openai-compatible` provider opencode uses does not consult
  llama-server's `n_ctx_train` or `/props.n_ctx`, so the wrapper injects
  the schema's `limit: { context, output }` block on its behalf.

## Laptop profile (RTX 5070 Ti, 12 GB)

For a Blackwell laptop GPU with 12 GB VRAM, `scripts/install-laptop.sh` +
`scripts/run-laptop.sh` set up **Qwen3.6-35B-A3B** (40 layers, MoE, 3B active)
from `mradermacher/Qwen3.6-35B-A3B-i1-GGUF` (`i1-Q4_K_S`, ~20 GiB):

- builds llama.cpp with `CUDA_ARCH=120` (Blackwell sm_120) against
  conda's `cuda-12.8.1` toolkit (12.4 doesn't ship sm_120 PTX),
- runs upstream master. The Qwen3.6-35B-A3B base model *does* ship an MTP head
  per Alibaba's model card (vLLM and SGLang both expose
  `--speculative-config '{"method":"mtp",...}'` for it), and upstream
  llama.cpp now has MTP support. The specific i1-Q4_K_S GGUF we use
  (`mradermacher/Qwen3.6-35B-A3B-i1-GGUF`) drops that head during conversion —
  `gguf-py` reports 0 tensors named `mtp*`/`nextn*` out of 733 total — so the
  laptop launcher intentionally omits `--spec-type draft-mtp` and focuses on
  upstream's `--n-cpu-moe` flag,
- `--n-cpu-moe 40` keeps every layer's MoE expert tensors on CPU (the experts
  are ~17 GiB and won't fit in 12 GiB VRAM); attention/router/embeddings stay
  on the GPU,
- **262 144 context** (the model's native max), `-fa on`, `-ctk q8_0 -ctv q8_0`
  (Q8_0 KV is near-lossless and halves the KV size vs fp16), `--cache-reuse
  256`, slot save/restore. The KV cache occupies ~2 720 MiB on the GPU at full
  context, leaving ~6.5 GiB free for compute buffers on a 12 GiB card.

#### Measured numbers (RTX 5070 Ti laptop, Qwen3.6-35B-A3B i1-Q4_K_S)

Server response timings via `llama-server`'s `/completion` endpoint
(`temperature=0`, `cache_prompt=false` unless noted), wrapped in the omp/opencode
launchers from `copilot-wrappers/`:

| prompt | decode | prefill (cold) | wall |
| ---: | ---: | ---: | ---: |
| ~50 tok | **56 tok/s** | 85 tok/s | 4.6 s for 256 toks |
| ~8 K  | **58 tok/s** | 1 173 tok/s | 7.9 s incl. prefill |
| ~32 K | **53 tok/s** | 1 306 tok/s | 24.7 s incl. prefill |
| ~128 K | **39 tok/s** | 1 027 tok/s | 125 s incl. prefill |

Decode stays well above the 25–45 tok/s estimate up to 32 K context; even at
128 K it is still usable for one-shot retrieval queries (a 13 K-token prompt
returns its answer in ~3.1 s end-to-end). Decode is RAM-bandwidth-bound by the
MoE-on-CPU split, so it scales with DDR5 speed/channels, not GPU clocks. Needs
≥ 32 GiB system RAM (48 GiB+ comfortable for full context).

### Why MoE-on-CPU is the right answer at 12 GB

Qwen3.6-35B-A3B is a sparse mixture-of-experts: 35 B total parameters, but only
**3 B active per token** (a router picks 8 of 256 experts per layer). Naive
quantized weight footprint at i1-Q4_K_S is ~20 GiB; ~17 GiB of that is expert
tensors that are touched sparsely, and ~3 GiB is the always-active stuff
(attention, embeddings, router, output head).

Trying to fit everything on a 12 GB card means heavy quantization (IQ2/IQ3)
that visibly hurts coding quality. The much better trade-off:

| Tensor class | Size | Where | Why |
|---|---:|---|---|
| Attention + router + embeddings | ~3 GiB | GPU | Touched every token; latency-sensitive |
| 262 K KV cache (Q8_0) | ~2.7 GiB | GPU | Touched every step; latency-sensitive |
| MoE experts (256 × 40 layers) | ~17 GiB | CPU RAM | Only 3 B params active per token; RAM-bandwidth bound |

The 3 B active params per token is small enough that DDR5 main-memory bandwidth
can deliver them in time, and the GPU only does the dense parts. Net effect:
you get 35B-quality output at ~30 tok/s on a laptop GPU you couldn't otherwise
fit even a 27B-Q4 model into.

### Walkthrough

**1. Install once on the laptop:**

```bash
git clone https://github.com/Maverobot/qwen36-mtp ~/Dev/qwen36-mtp
cd ~/Dev/qwen36-mtp
bash scripts/install-laptop.sh
```

This installs miniconda's CUDA 12.8.1 toolkit into a dedicated env, builds
llama.cpp with `CUDA_ARCH=120`, downloads the GGUF (~20 GiB) via `hf` +
`hf_transfer`, and writes `~/.config/qwen36-mtp/laptop.env` with `LLAMA_BIN`
and `MODEL_PATH` filled in.

**2. Run the server:**

```bash
set -a; source ~/.config/qwen36-mtp/laptop.env; set +a
~/Dev/qwen36-mtp/scripts/run-laptop.sh
```

Listens on `127.0.0.1:8080`, OpenAI-compatible at `/v1/chat/completions`. Test:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"pong?"}]}'
```

**3. Common overrides** (set the env var before running):

| Var | Default | When to change |
|---|---|---|
| `CTX_SIZE` | `131072` | Drop to `65536` if you OOM on KV; bump to `262144` only with ≥ 64 GiB RAM |
| `N_CPU_MOE` | `40` | **Lower** (e.g. 36, 32) if you have spare VRAM — moves more experts onto GPU = faster decode |
| `PORT` | `8080` | If 8080 is busy |
| `SLOT_CACHE_DIR` | unset | Set to e.g. `~/.cache/qwen36-laptop/slots` to persist KV across restarts |
| `PARALLEL` | `1` | Bump to 2+ for concurrent agent slots (costs VRAM) |

Tuning loop: watch `nvidia-smi` while serving — if VRAM stays well below 11 GiB
during steady-state decode you're leaving performance on the table; lower
`N_CPU_MOE` by 4 and rerun until you're at ~10.5 GiB.

**4. Optional — run as a systemd user service:**

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/qwen36-laptop.service <<'EOF'
[Unit]
Description=Qwen3.6-35B-A3B (laptop)
After=network.target

[Service]
EnvironmentFile=%h/.config/qwen36-mtp/laptop.env
ExecStart=%h/Dev/qwen36-mtp/scripts/run-laptop.sh
Restart=on-failure

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now qwen36-laptop
journalctl --user -fu qwen36-laptop
```

The omp/opencode wrappers above work unchanged against the laptop server —
just start the laptop server on `:8080` and the wrappers' built-in proxies
will spawn on `:8091`/`:8092` as usual.

## Thinking mode

Qwen3.6 thinks by default. The launcher sets `--reasoning-format deepseek`, so
`<think>` blocks land in the response's separate `reasoning_content` field —
clients that don't render reasoning still see clean `content`. To disable
thinking per request:

```json
{ "chat_template_kwargs": { "enable_thinking": false } }
```

For most coding agent workflows, leave thinking on; MTP keeps the latency cost
modest.

## Caveats

- The build links against the conda CUDA env via rpath. If you delete or rename
  that env, set `LD_LIBRARY_PATH` to a CUDA 12 lib dir before launching.
- Don't use `aria2c` for the GGUF download — silent corruption has been
  reported for multi-GB transfers. The installer uses `hf` (huggingface-hub) +
  `hf_transfer` instead.
- 262 144 ctx leaves only ~700 MB VRAM headroom; risky if anything else
  touches the GPU. 131 072 is the recommended default; raise to 196 608 only
  when you need the extra context and can spare the VRAM.
- `llama-server` tool-calling via `--jinja` works for simple agents. For
  multi-tool / strict-schema workloads, put **LiteLLM** in front of it.
- This recipe is dense-27B-specific. A 35B variant would need its own GGUF and
  possibly a different `CTX_SIZE`.
- Upstream MTP uses `--spec-type draft-mtp`. `scripts/run.sh` enables it with
  `PARALLEL=2` by default. `scripts/run-multi.sh` / `qwen36-multi.service` are
  deprecated compatibility shims for old installs.

## Credits

- Upstream MTP support: [`ggml-org/llama.cpp#22673`](https://github.com/ggml-org/llama.cpp/pull/22673)
- MTP-preserving GGUF: [`localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF`](https://huggingface.co/localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF)
- Reference recipe + 100 tok/s claim: noonghunna's `qwen36-27b-single-3090` writeup

## License

MIT
