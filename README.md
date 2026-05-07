# qwen36-mtp

High-performance local serving of **Qwen3.6 with Multi-Token Prediction (MTP)**
on a single 24 GB consumer GPU (tested on RTX 4090; should also work on RTX 3090
with `CUDA_ARCH=86`).

The point of this repo is to capture the *exact* recipe — patched
[llama.cpp](https://github.com/ggerganov/llama.cpp) fork, MTP-preserving GGUF,
CUDA-toolkit-via-conda, shared-cudart link fix, sane launch flags — so that a
fresh box can go from zero to a working OpenAI-compatible endpoint with one
command.

## Why this stack?

| Need | What this repo gives you |
| --- | --- |
| Real Qwen3.6 quality on 24 GB | dense 27B (or 35B-class with the right GGUF), q4 weights, q4 KV cache |
| 100+ tok/s decode | NextN/MTP speculative decoding (~3× over no-MTP) |
| Very large context (≥196K) | hybrid GatedDeltaNet + GatedAttention layout (only 16/64 layers carry softmax KV) |
| Strong prefix caching | `--cache-idle-slots`, `--ctx-checkpoints`, KV-slot save/restore baked into the fork |
| OpenAI-compatible | `llama-server` with `--jinja` and `--reasoning-format deepseek` |
| Tool-calling | `--jinja` enables Qwen's tool-call template; LiteLLM in front recommended for heavy agents |

## Measured numbers (RTX 4090, this script)

| ctx | decode | prefill | MTP accept | VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 32 768 | 124 tok/s | 310 tok/s | 76.8 % | 18.8 GB |
| 196 608 | 142 tok/s | 196 tok/s | 4/4 | 22.2 GB |
| 262 144 | 159 tok/s | 342 tok/s | 4/4 | 23.8 GB |

(HF model card reports 100.3 tok/s short-ctx on a 3090 Ti with the same recipe.)

### Prefill-cache TTFT (RTX 4090, 196 608 ctx, 3 628-token prompt)

| call | wall | `prompt_n` re-prefilled | `cache_n` reused via KV-shift |
| --- | ---: | ---: | ---: |
| cold | **3.11 s** | 3628 | 0 |
| warm (same preamble, different tail) | **0.42 s** | 515 | 3112 |
| warm again | **0.38 s** | 514 | 3112 |

~7.5× faster TTFT on subsequent calls thanks to `--cache-reuse 256`,
`--cache-idle-slots`, and the patched fork's KV-slot persistence.

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
| `--slot-save-path` (+ patched fork) | auto-save/restore slot KV to disk; survives `systemctl restart qwen36` |
| `--ctx-checkpoints 8 --checkpoint-every-n-tokens 2048` | mid-prefill snapshots so partial cancels don't waste work |
| `--cache-ram -1 --kv-unified` | unlimited RAM-side overflow + the unified-KV mode required by the above |

## Install

Prereqs: NVIDIA driver (CUDA 12-capable), `git`, `cmake`, `conda` (Miniconda or
Mamba). Nothing else. The installer fetches its own CUDA toolkit into a conda
env, so a "driver-only" Linux box works.

```bash
git clone https://github.com/Maverobot/qwen36-mtp.git
cd qwen36-mtp
./scripts/install.sh
```

Override knobs via env vars (defaults are tuned for a single RTX 4090 with 24 GB):

```bash
PREFIX=$HOME/llm/qwen36 \
CTX_SIZE=131072 \
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

- `$PREFIX/llama.cpp/build/bin/llama-server` — patched binary
- `$PREFIX/models/qwen36-27b-mtp/Qwen3.6-27B-MTP-IQ4_XS.gguf` — ~14 GiB
- `$PREFIX/slot-cache/` — on-disk slot KV cache (auto-save/restore across restarts)
- `~/.config/qwen36-mtp/env` — runtime config (paths, ctx size, port, …)
- `~/.config/systemd/user/qwen36.service` — auto-start unit; ExecStart points
  at `scripts/run.sh` *in this repo* so `git pull` updates the launcher.

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

### Two profiles

The repo ships two launchers and matching systemd units:

| Profile        | Launcher              | Unit                  | Port | MTP | Slots | When to use                                      |
|----------------|-----------------------|-----------------------|-----:|:---:|:----:|--------------------------------------------------|
| **MTP single** | `scripts/run.sh`      | `qwen36.service`      | 8080 |  ✓  |  1   | Single-stream coding; max tok/s (~100+)          |
| **multi**      | `scripts/run-multi.sh`| `qwen36-multi.service`| 8081 |  ✗  |  4   | Concurrent agents/subagents (e.g. Copilot CLI)   |

```bash
systemctl --user start qwen36         # MTP profile on :8080
systemctl --user start qwen36-multi   # multi profile on :8081
```

Measured single-stream tok/s on RTX 4090:
- MTP profile (`:8080`): ~83–106 tok/s
- multi profile (`:8081`): ~37 tok/s/stream × 4 streams = ~150 tok/s aggregate

You can run them simultaneously (different ports, different VRAM pools) only
if you have headroom. With 196 608 ctx each they will not co-exist on a 24 GB
card; pick one at a time.

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

Run them with no setup once the server is up:

```bash
omp-qwen36-27b -p "Reply with one word: pong"
opencode-qwen36-27b run "Reply with one word: pong"
```

The omp wrapper also writes `~/.omp/agent/models.yml` with `api: openai-completions`,
`auth: none`, `discovery.type: llama.cpp` (omp's discovered llama.cpp provider
hard-codes `api: openai-responses` which doesn't match `/chat/completions`), and
busts omp's sqlite `model_cache` if a stale `baseUrl` was cached. The opencode
wrapper uses `OPENCODE_CONFIG_DIR=~/.config/qwen36-mtp/opencode` so it does not
touch your normal `~/.config/opencode/opencode.json`.

## Laptop profile (RTX 5070 Ti, 12 GB)

For a Blackwell laptop GPU with 12 GB VRAM, `scripts/install-laptop.sh` +
`scripts/run-laptop.sh` set up **Qwen3.6-35B-A3B** (40 layers, MoE, 3B active)
from `mradermacher/Qwen3.6-35B-A3B-i1-GGUF` (`i1-Q4_K_S`, ~20 GiB):

- builds llama.cpp with `CUDA_ARCH=120` (Blackwell sm_120) against
  conda's `cuda-12.8.1` toolkit (12.4 doesn't ship sm_120 PTX),
- runs upstream master rather than the `crucible-mtp` fork (Qwen3.6-35B-A3B has
  no MTP head; we want upstream's `--n-cpu-moe` flag instead),
- `--n-cpu-moe 40` keeps every layer's MoE expert tensors on CPU (the experts
  are ~17 GiB and won't fit in 12 GiB VRAM); attention/router/embeddings stay
  on the GPU,
- 128 K context, `-fa on`, `-ctk q8_0 -ctv q8_0` (Q8_0 KV is near-lossless and
  halves the KV size vs fp16), `--cache-reuse 256`, slot save/restore.

Realistic numbers on a 5070 Ti laptop with DDR5-5600+ RAM: ~25–45 tok/s decode
(RAM-bandwidth-bound because of the MoE-on-CPU split), 250–400 tok/s prefill.
Needs ≥ 32 GiB system RAM (48 GiB+ comfortable for 128 K ctx).

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
  touches the GPU. 196 608 is the recommended default.
- `llama-server` tool-calling via `--jinja` works for simple agents. For
  multi-tool / strict-schema workloads, put **LiteLLM** in front of it.
- This recipe is dense-27B-specific. A 35B variant would need its own GGUF and
  possibly a different `CTX_SIZE`.
- **MTP and multi-slot are mutually exclusive** in the `crucible-mtp` fork. The
  server refuses to start with `--spec-type mtp` and `--parallel >1`
  (`MTP currently supports only n_parallel=1`). `scripts/run.sh` handles this
  automatically: if you set `PARALLEL=2+` (or `DISABLE_MTP=1`) in the env file,
  it drops `--spec-type mtp` and enables `--parallel N --kv-unified
  --slot-prompt-similarity $SPS` for concurrent clients (e.g. Copilot CLI
  subagents) at the cost of ~30–50% lower single-stream tok/s.

## Credits

- Patched llama.cpp fork: [`nickstx/llama.cpp#crucible-mtp`](https://github.com/nickstx/llama.cpp/tree/crucible-mtp)
- MTP-preserving GGUF: [`localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF`](https://huggingface.co/localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF)
- Reference recipe + 100 tok/s claim: noonghunna's `qwen36-27b-single-3090` writeup

## License

MIT
