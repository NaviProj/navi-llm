# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`navi-llm` is a local LLM inference crate built on `llama-cpp-2` (Rust bindings for llama.cpp). It loads GGUF models and provides text completion, multi-turn chat with KV Cache reuse, tool calling via grammar-constrained generation, and vision (image+text) inference via multimodal projectors (mmproj).

## Build & Run

```bash
cargo build -p navi-llm                    # Build
cargo build -p navi-llm --features metal   # macOS Metal acceleration
cargo build -p navi-llm --features cuda    # CUDA acceleration

# Test binaries (require a GGUF model file)
cargo run --bin llm_test -p navi-llm
cargo run --bin session_test -p navi-llm
cargo run --bin tool_call_demo -p navi-llm
cargo run --bin vllm_test -p navi-llm
cargo run --bin vision_test -p navi-llm    # Requires model + mmproj file

cargo test -p navi-llm
cargo clippy -p navi-llm
```

## Architecture

Three layers of abstraction, from simple to full-featured:

1. **`LlmModel`** (`model.rs`) — Single-shot completions. Loads a model, creates a fresh context per call, runs tokenize → decode → sample loop. Supports streaming via callback. No state between calls.

2. **`LlmSessionFactory` / `ManagedSession`** (`session.rs`) — Multi-turn chat with KV Cache reuse. The factory holds the model; each `ManagedSession` owns a `LlamaContext` and maintains conversation history. Key mechanism: **incremental encoding** — on each turn, it diffs new tokens against `encoded_tokens` to find the common prefix, only encodes the delta, and partially clears KV Cache when history diverges. Supports tool calling via `set_tools_json()` which enables grammar-constrained sampling and OAI-compatible tool call parsing.

3. **`VisionLlmModel`** (`vision.rs`) — Standalone multimodal inference. Loads both a language model and an mmproj (vision projector) via `MtmdContext`. Accepts image files or raw bytes, tokenizes text+image chunks together, then runs the standard sample loop.

`LlmSessionFactory` also has integrated vision support (`complete_with_image_bytes_streaming`) when configured with `mmproj_path`.

### Key implementation details

- **Chat templates**: Uses `apply_chat_template_oaicompat` with Jinja rendering. Falls back to `chatml` template if the model's built-in template fails.
- **Thinking mode**: `enable_thinking` config controls whether models like Qwen3.5 emit `<think>` reasoning blocks.
- **Grammar/tool calling**: When `tools_json` is set on a session, `build_prompt` returns `ChatTemplateResult` with grammar rules and trigger patterns. The session constructs a lazy grammar sampler that activates on tool-call triggers.
- **Cancellation**: `ManagedSession` supports external cancellation via `Arc<AtomicBool>` checked each token.
- **All prompt construction** goes through `build_prompt()` which serializes messages to OpenAI-compatible JSON format.

### Config (`config.rs`)

`LlmConfig` uses a builder pattern: `LlmConfig::new(path).with_ctx_size(4096).with_max_tokens(512).with_system_prompt("...")`. Defaults: ctx_size=2048, max_tokens=1024, seed=1234, enable_thinking=true.
