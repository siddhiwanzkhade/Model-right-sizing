# Why Does Quantization Slow Down LLMs? A Kernel-Level Study

I wanted to understand why quantization slows down inference on certain hardware — not just measure that it does. So I benchmarked Qwen2.5-0.5B-Instruct across FP16, INT8, and INT4 on a Tesla T4 GPU, then profiled the exact CUDA kernels underneath each configuration to trace the root cause.

---

## Results

| Precision | Throughput | Decode Latency | Peak Memory |
|-----------|------------|----------------|-------------|
| FP16      | 28.6 tok/s | 35.1 ms/tok    | 2.87 GB     |
| INT8      | 6.1 tok/s  | 162.7 ms/tok   | 2.54 GB     |
| INT4      | 12.9 tok/s | 77.2 ms/tok    | 2.39 GB     |

Memory savings were real. But FP16 was the fastest — by 4.6x over INT8 and 2.2x over INT4.

---

## What the Profiler Revealed

**FP16** — one kernel per linear layer: `aten::mm`, a clean matrix multiply. Total CUDA time: 388ms.

**INT8** — four kernels per linear layer, every single time:
`quantize activations → compute → dequantize result → cast dtype`
The model has 168 linear layers. That's 4× more kernel launches than FP16. Total CUDA time: 894ms.

**INT4** — the kernel name said it all: `kgemm_4bit_inference_naive`. Naive means bitsandbytes has no optimized path for INT4 on this hardware and falls back to a generic implementation. 27.8% of its total CUDA time was just dtype casting.

> **Why this happens:** bitsandbytes stores weights at lower precision but converts them back to FP16 before every matrix multiply — paying a dequantization cost on every layer, every token, every forward pass. This overhead outweighs the memory bandwidth savings at inference-time batch sizes.

---

## The Insight

Quantization is a hardware-software co-design problem. The overhead isn't in the math — it's in the kernel structure bitsandbytes uses to work around missing low-bit hardware support. On hardware with native low-bit compute primitives, this overhead disappears entirely.

**Next:** Re-running on A100 to verify — if INT8 flips from slower to faster on Ampere, it confirms that kernel support, not precision alone, determines whether quantization helps or hurts.

---

`Qwen2.5-0.5B · bitsandbytes · torch.profiler · Tesla T4 · Google Colab`
