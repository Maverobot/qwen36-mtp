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

## Install

Prereqs: NVIDIA driver (CUDA 12-capable), `git`, `cmake`, `conda` (Miniconda or
Mamba). Nothing else. The installer fetches its own CUDA toolkit into a conda
env, so a "driver-only" Linux box works.

```bash
git clone https://github.com/Maverobot/qwen36-mtp.git
cd qwen36-mtp
./scripts/install.sh
```

Override knobs via env vars:

```bash
PREFIX=$HOME/llm/qwen36 \
CTX_SIZE=131072 \
PORT=8081 \
CUDA_ARCH=86 \                 # 3090 / 3090 Ti
./scripts/install.sh
```

The script creates:

- `$PREFIX/llama.cpp/build/bin/llama-server` — patched binary
- `$PREFIX/models/qwen36-27b-mtp/Qwen3.6-27B-MTP-IQ4_XS.gguf` — ~14 GiB
- `$PREFIX/run-qwen36.sh` — launcher (edit this for tuning)
- `~/.config/systemd/user/qwen36.service` — optional auto-start unit

## Run

```bash
~/Dev/qwen36/run-qwen36.sh
# or, as a systemd user service:
systemctl --user daemon-reload
systemctl --user enable --now qwen36
sudo loginctl enable-linger $USER     # keep running across logout
```

Endpoint: `http://localhost:8080/v1` (model alias `qwen3.6-27b`).

```bash
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"reply: pong"}]}'
```

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

## Credits

- Patched llama.cpp fork: [`nickstx/llama.cpp#crucible-mtp`](https://github.com/nickstx/llama.cpp/tree/crucible-mtp)
- MTP-preserving GGUF: [`localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF`](https://huggingface.co/localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF)
- Reference recipe + 100 tok/s claim: noonghunna's `qwen36-27b-single-3090` writeup

## License

MIT
