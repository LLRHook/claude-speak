# Architecture Decisions

## ADR-001: Local Speech-to-Text Backend

**Date:** 2026-02-26
**Status:** Proposed
**Context:** Replace Superwhisper (paid macOS app) with a built-in STT solution so that claude-speak has zero paid-app dependencies. The replacement must run entirely locally, work on Apple Silicon, be callable from Python, and stay under 500 ms for a 5-second utterance.

---

### Candidates Evaluated

#### 1. pywhispercpp (Python bindings for whisper.cpp)

- **PyPI package:** `pywhispercpp`
- **Latest version:** 1.4.1 (released 2025-12-30)
- **Apple Silicon optimization:** Supports CoreML (Apple Neural Engine) and Metal GPU via compile-time flags. CoreML encoder acceleration requires building from source with `WHISPER_COREML=1 pip install .`. Pre-built PyPI wheels run CPU-only; Metal (via GGML) is enabled in standard builds. CoreML gives a documented **>3x speedup** over CPU-only on M-series chips. First CoreML run is slow because the ANE compiles the model to a device-specific blob; subsequent runs use the cached blob.
- **Model sizes:** tiny (39M params, ~75 MB), base (74M, ~142 MB), small (244M, ~466 MB), medium (769M, ~1.5 GB), large-v3/turbo (~1.5 GB). Models auto-downloaded on first use from Hugging Face / whisper.cpp CDN.
- **Latency (5-second audio):** No direct 5-second benchmark found. Using the Mac M4 study as a proxy: `whisper-tiny` processes 10 seconds in **0.37 s** (RTF ≈ 27x faster than real-time); `base` takes **0.54 s** for 10 seconds (RTF ≈ 18x). Extrapolating linearly, a 5-second clip would finish in roughly **0.19 s (tiny) / 0.27 s (base)** on M4 with CPU. CoreML adds a further 3x multiplier, so a CoreML build would be substantially faster. These values comfortably beat the 500 ms target even without CoreML.
- **Memory:** Tiny ~0.5 GB RAM at runtime; base ~0.6 GB; small ~1 GB; medium ~2 GB.
- **API simplicity:** Three-line integration. `Model` constructor accepts model name (auto-downloaded) or local path; `model.transcribe(audio_path)` returns a list of segments. Segment callbacks enable real-time streaming output.
- **Maintenance:** Active. Repository (absadiki/pywhispercpp) released 1.4.1 on 2025-12-30. Tracks the upstream ggml-org/whisper.cpp which is among the most actively maintained open-source AI projects (2k+ commits).
- **Known issues:** CoreML build requires compiling from source; the PyPI wheel is CPU-only. First CoreML inference on a new device takes several minutes for ANE model compilation. Models up to `large-v2` are supported by CoreML; `large-v3` CoreML support has had historical issues. Metal (non-CoreML GPU) works in standard builds and is significantly faster than pure CPU on Apple Silicon without requiring source compilation.

---

#### 2. faster-whisper (CTranslate2 backend)

- **PyPI package:** `faster-whisper`
- **Latest version:** 1.2.1 (released 2025-10-31)
- **Apple Silicon optimization:** None. faster-whisper uses CTranslate2 which requires CUDA (NVIDIA) for GPU acceleration. On Apple Silicon it runs CPU-only. There is no Metal, CoreML, or MPS path. This is a confirmed and unresolved limitation documented in the project's own issue tracker (issue #325, discussion #1227).
- **Model sizes:** Same Whisper model lineup (tiny through large-v3, distil-large-v3). Auto-downloaded from Hugging Face.
- **Latency (5-second audio):** CPU-only on Apple Silicon is significantly slower than accelerated alternatives. In the mac-whisper-speedtest benchmark on M4 Pro, faster-whisper scored the worst of all implementations at **6.96 s average** (for the full test audio), roughly 36x slower than the best CoreML option. For a 5-second clip this likely means 1–3 s depending on model size — probably within the 500 ms budget for tiny/base only, but without headroom.
- **Memory:** Similar to other Whisper variants: tiny ~1 GB, base ~1 GB, small ~2 GB, medium ~5 GB.
- **API simplicity:** Clean. `WhisperModel("base", device="cpu")` + generator-style segment iteration. Well-documented.
- **Maintenance:** Active. 1.2.1 released 2025-10-31. Well-maintained by SYSTRAN.
- **Known issues:** No GPU acceleration on Apple Silicon is a fundamental architectural constraint, not a bug. This makes it the slowest option for macOS M-series users. The 4x CPU speed advantage it has over openai/whisper on Intel/NVIDIA hardware is partially negated on Apple Silicon because competitors use the Neural Engine or Metal.

---

#### 3. mlx-whisper (Apple MLX framework)

- **PyPI package:** `mlx-whisper`
- **Latest version:** 0.4.3 (released 2025-08-29)
- **Apple Silicon optimization:** Native. MLX is Apple's own open-source ML framework, designed exclusively for Apple Silicon. It uses the GPU (via Metal) and leverages the unified memory architecture — no CUDA dependency at all. mlx-whisper runs the full Whisper model on the GPU with zero-copy CPU/GPU memory sharing.
- **Model sizes:** Same family (tiny through large-v3-turbo), sourced from the `mlx-community` Hugging Face collection. Automatically downloaded on first use via Hugging Face Hub. Default model is `mlx-community/whisper-tiny` (74 MB). Recommended balanced model: `mlx-community/whisper-large-v3-turbo` (~800 MB) or `mlx-community/distil-whisper-large-v3` (1.5 GB). Quantized variants (4-bit, 8-bit) are available to reduce VRAM pressure.
- **Latency (5-second audio):** No dedicated 5-second benchmark found. In comparative benchmarks on M4 Pro hardware, mlx-whisper achieved **1.02 s average** over the full test suite (all model sizes), placing it mid-table. For tiny/base specifically, the RTF extrapolations from Mac M4 testing (27x and 18x real-time respectively) suggest that mlx-whisper tiny/base would transcribe a 5-second clip in well under 500 ms. For larger models like large-v3-turbo, reported figures are ~63 s to process 10 minutes of audio (RTF ≈ 0.1), meaning a 5-second clip would take roughly **3–5 s** — too slow for interactive use. The correct model choice for the 500 ms target is tiny, base, or small.
- **Memory:** Runs in the Apple unified memory pool. Tiny ~0.5 GB; base ~0.6 GB; small ~1 GB; large-v3-turbo ~4–5 GB depending on quantization. The Activity Monitor GPU indicator reaches ~90% during inference, confirming genuine GPU utilization.
- **API simplicity:** Extremely simple. `mlx_whisper.transcribe(audio_path)["text"]` is the entire API. Model is specified as a Hugging Face repo ID string; auto-download is transparent. No class instantiation required for basic use; model is loaded on first call and cached. Closest to a drop-in replacement.
- **Maintenance:** Active. Published by Apple ML Research / ml-explore team. PyPI version 0.4.3 released 2025-08-29. The parent `mlx` framework and `mlx-examples` repository receive continuous updates. Apple announced MLX publicly at WWDC 2023 and has continued funding it as a first-party research framework — low abandonment risk.
- **Known issues:** macOS-only (by design — MLX does not exist on Windows/Linux). First invocation downloads the model from Hugging Face; requires internet on first run. Model is not pre-loaded between calls unless the caller manages caching explicitly, which adds ~0.2–1 s per-call startup on subsequent invocations if the Python process is fresh. The `mlx-whisper` package version number is still in the 0.x series, indicating pre-1.0 API stability, though in practice the API has been stable.

---

#### 4. ONNX-based Whisper (onnx-asr / onnxruntime)

- **PyPI package:** `onnx-asr` (primary candidate); also `sherpa-onnx`
- **Latest version:** onnx-asr 0.10.2 (released 2026-01-18)
- **Apple Silicon optimization:** Partial. ONNX Runtime ships arm64 macOS wheels and supports the CoreML Execution Provider, which routes ops to the ANE. However, the CoreML EP for ONNX Runtime has historically lagged behind whisper.cpp's CoreML integration in both coverage and stability. The `onnxruntime-silicon` PyPI package (a drop-in with CoreML enabled) was a community-maintained shim; the official `onnxruntime` now ships arm64 binaries but CoreML integration quality is variable.
- **Model sizes:** Whisper models must be separately converted to ONNX format or downloaded from community ONNX exports. Not all sizes have stable, widely-distributed pre-built ONNX checkpoints. This adds friction compared to the other candidates.
- **Latency:** No Apple Silicon-specific benchmarks found for onnx-asr or similar ONNX Whisper Python packages. In general, ONNX Runtime on Apple Silicon CPU is comparable to faster-whisper (both use optimized C++ inference engines), but CoreML-accelerated ONNX would be faster. Benchmark data is sparse enough that no firm number can be cited.
- **Memory:** Similar to other Whisper variants per model size.
- **API simplicity:** Higher friction. Requires sourcing or converting ONNX model files, selecting the right runtime package, and configuring the execution provider. The `onnx-asr` package aims to simplify this but is a smaller ecosystem than the other three candidates.
- **Maintenance:** `onnx-asr` is actively developed (0.10.2, 2026-01-18). However, the ecosystem around ONNX Whisper on macOS is more fragmented than the other options. CoreML ONNX support is maintained by Microsoft but receives lower community attention for macOS than CUDA.
- **Known issues:** Model distribution is fragmented — no single canonical source for all model sizes in ONNX format. CoreML Execution Provider has known limitations with dynamic input shapes. Overall, this path requires the most manual setup for the least well-documented Apple Silicon performance profile.

---

### Comparison Table

| Criterion | pywhispercpp 1.4.1 | faster-whisper 1.2.1 | mlx-whisper 0.4.3 | onnx-asr 0.10.2 |
|---|---|---|---|---|
| Apple Silicon acceleration | Metal (standard) / CoreML (source build) | None — CPU only | Native MLX / Metal GPU | CoreML EP (partial) |
| Auto-download models | Yes | Yes | Yes (Hugging Face) | Partial (depends on provider) |
| Tiny model 5s latency (est.) | ~0.19 s (CPU) / ~0.06 s (CoreML) | ~0.5–1 s (CPU) | ~0.2 s (GPU) | Unknown |
| Base model 5s latency (est.) | ~0.27 s (CPU) / ~0.09 s (CoreML) | ~0.8–1.5 s (CPU) | ~0.28 s (GPU) | Unknown |
| Recommended model for <500 ms | tiny or base | tiny only (risky) | tiny, base, or small | Unknown |
| Memory (tiny/base/small) | 0.5 / 0.6 / 1 GB | 0.5 / 0.6 / 1 GB | 0.5 / 0.6 / 1 GB | ~same |
| API lines for basic use | 3 | 4 | 1 | 4+ (plus model setup) |
| Installation complexity | pip install (CPU); source build for CoreML | pip install | pip install | pip install + model sourcing |
| Latest release | 2025-12-30 | 2025-10-31 | 2025-08-29 | 2026-01-18 |
| Maintenance health | Strong (tracks active whisper.cpp) | Strong | Strong (Apple-backed) | Moderate |
| macOS-only risk | No (cross-platform) | No | Yes (MLX = Apple Silicon only) | No |

Latency estimates for pywhispercpp are derived from Mac M4 benchmarks (tiny: 0.37 s for 10 s audio; base: 0.54 s for 10 s audio) and scaled to 5 s. CoreML estimates apply a conservative 3x speedup factor. mlx-whisper estimates are based on comparable RTF figures from mlx-whisper benchmarks on M4 hardware. faster-whisper CPU estimates are derived from the mac-whisper-speedtest relative ranking showing it 36x slower than the fastest CoreML option.

---

### Recommendation: mlx-whisper as default, pywhispercpp as fallback

**Default backend: `mlx-whisper`**

mlx-whisper is the best fit for this project given the following:

1. **Genuine GPU acceleration out of the box.** Unlike faster-whisper (CPU-only on Apple Silicon) and unlike pywhispercpp (CoreML requires a source build), mlx-whisper accelerates automatically via Metal on every Apple Silicon Mac without any extra installation steps. `pip install mlx-whisper` and the first call to `mlx_whisper.transcribe()` is already GPU-accelerated.

2. **Simplest API.** The single-function API (`mlx_whisper.transcribe(path)["text"]`) is the lowest-friction integration point. There is no model class, no device configuration, and no compute type to select. This minimizes the code surface in `voice_input.py`.

3. **Meets the latency target on the right model.** With tiny or base models, a 5-second utterance transcribes in well under 500 ms on M2+ hardware. The recommended default is `mlx-community/whisper-base` (or `whisper-large-v3-turbo` for users who prefer higher accuracy and have M3/M4 hardware). The turbo model at RTF ~0.1 would process a 5-second clip in roughly 0.5 s on M2 — borderline, so tiny/base should be the shipping default.

4. **Auto-download works transparently.** Models are fetched from the `mlx-community` Hugging Face organization the first time they are needed, then cached in `~/.cache/huggingface/hub/`. No manual download step required.

5. **Apple-backed longevity.** MLX is a first-party Apple framework maintained by Apple ML Research. The risk of abandonment is lower than for community-maintained bindings.

**Fallback / alternative: `pywhispercpp`**

pywhispercpp should be treated as the secondary option for:

- Users who want to run on non-Apple-Silicon macOS (unlikely given the project's existing macOS-only stance, but possible on older Intel Macs).
- Users willing to compile from source for CoreML to extract the maximum speed from the Neural Engine.
- Situations where mlx is unavailable (e.g., a future Linux port of the voice input pipeline).

The pywhispercpp API is nearly as simple, and the Metal-accelerated standard build still comfortably beats the 500 ms target on tiny/base models even without CoreML.

**Ruled out:**

- **faster-whisper:** Eliminated because it has no GPU path on Apple Silicon. CPU-only inference is the slowest option in this comparison and provides no Apple Silicon advantage over the other candidates. The killer feature of faster-whisper (CUDA quantized inference) is simply unavailable on M-series chips.
- **onnx-asr:** Eliminated due to fragmented model distribution, unclear Apple Silicon benchmark data, and higher integration complexity relative to mlx-whisper and pywhispercpp. The CoreML ONNX path exists but is less mature and less documented than the whisper.cpp CoreML path.

---

### Integration Notes (for the implementation phase)

- The `[input]` section of `claude-speak.toml` should gain a `stt_backend` key (default: `"mlx-whisper"`) and `stt_model` key (default: `"mlx-community/whisper-base"`).
- The new `voice_input.py` STT path should load the model once at daemon start (to absorb the Hugging Face download + model-load latency) and keep it in memory, rather than loading per-utterance.
- Audio capture should continue to use the existing microphone pipeline (after wake word hands off the mic) and produce a WAV file (16 kHz, mono, 16-bit PCM) as input to `mlx_whisper.transcribe()`.
- The Superwhisper integration in `voice_input.py` can be retained behind the existing `[input] superwhisper = true` flag so that users who prefer Superwhisper are not affected.
