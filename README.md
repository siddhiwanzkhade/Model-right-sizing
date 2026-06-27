# Model-right-sizing

## When Quantization Does Not Speed Up LLMs....

I did this experiment to understand **whether quantization actually improves LLM inference speed on real hardware**, or whether it mainly reduces memory while introducing runtime overhead.

I profiled **Qwen2.5-0.5B-Instruct** on a **Tesla T4 GPU** to study a common ML systems assumption:

> Lower precision should reduce memory and make inference faster.

I benchmarked the same model across:

* FP16
* INT8
* INT4

I measured both phases of LLM inference:

* **Prefill / Time to First Token (TTFT):** how long the model takes to start responding
* **Decode latency:** how fast the model generates each new token

## What I Observed

The memory results were expected:

* INT4 used the least memory
* INT8 used less memory than FP16
* FP16 used the most memory

But latency told a different story.

| Precision |   TTFT | Decode Latency |   Throughput | Peak Memory |
| --------- | -----: | -------------: | -----------: | ----------: |
| FP16      |  40 ms |  35.1 ms/token | 28.6 tok/sec |     2.87 GB |
| INT8      | 203 ms | 162.7 ms/token |  6.1 tok/sec |     2.54 GB |
| INT4      | 105 ms |  77.2 ms/token | 12.9 tok/sec |     2.39 GB |

FP16 was consistently faster than both quantized versions.

## Deeper Analysis

I then tested the model across:

* different input prompt lengths
* different output generation lengths
* different batch sizes

The same pattern held:

* TTFT increased as prompts got longer
* decode latency stayed mostly stable across output lengths
* throughput improved with batch size
* FP16 still remained the fastest overall

INT4 became more competitive at longer prompts and larger batches, but it still did not outperform FP16 on this setup.

## Key Insight

This was a useful reminder that **smaller models are not always faster models**.

For this small LLM on Tesla T4, quantization reduced memory footprint but introduced enough runtime overhead that latency became worse. The likely reason is that FP16 benefits from optimized GPU execution on T4, while INT4/INT8 may introduce low-bit kernel, casting, or dequantization overhead.

## Takeaway

Model right-sizing is not just about model size or precision.

Real deployment decisions need to measure:

* memory
* latency
* throughput
* prefill behavior
* decode behavior
* target hardware behavior

In this experiment, quantization was useful for memory reduction, but not for latency improvement.

The broader lesson:

> LLM inference latency is not determined by precision alone. Runtime kernels, hardware support, model size, and inference phase matter.

#####
# When Quantization Slows Down LLMs — An ML Systems Study

I ran into a counterintuitive ML systems result last week. I was benchmarking Qwen2.5-0.5B-Instruct across three precisions — FP16, INT8, INT4 — on a Tesla T4 GPU, expecting the standard result: lower precision, less memory, faster inference. The memory results were exactly right. But latency went the wrong direction entirely — FP16 was the fastest, and the quantized versions were slower. So I profiled the exact CUDA kernels underneath to find out why.

---

## Results

| Precision | Throughput | Decode Latency | Peak Memory |
|-----------|------------|----------------|-------------|
| FP16      | 28.6 tok/s | 35.1 ms/tok    | 2.87 GB     |
| INT8      | 6.1 tok/s  | 162.7 ms/tok   | 2.54 GB     |
| INT4      | 12.9 tok/s | 77.2 ms/tok    | 2.39 GB     |

---

## Why — The Kernel-Level Explanation

> **CUDA Kernel:** A function that runs on the GPU. Every operation — a matrix multiply, a dtype cast, an activation — is a separate kernel the CPU launches onto the GPU.

> **Kernel Launch Overhead:** Each launch costs ~5–50μs of setup time regardless of how much work the kernel does. Many small kernels means overhead dominates over actual computation.

**FP16** runs one kernel per linear layer: `aten::mm` — a clean matrix multiply on T4's native FP16 Tensor Cores. Total CUDA time: 388ms.

**INT8** runs four kernels per linear layer: quantize activations → compute → dequantize result → cast dtype. Across 168 linear layers, that's 4× more kernel launches. Total CUDA time: 894ms.

**INT4** — the profiler said it best. The kernel is literally named `kgemm_4bit_inference_naive`. Naive, because T4 has no native INT4 hardware path. 27.8% of its CUDA time was just dtype casting.

> **Dequantization Tax:** bitsandbytes stores weights at lower precision but converts them back to FP16 before every matrix multiply — paying a conversion cost on every layer, every token, every forward pass.

---

## The Insight

**Quantization is a hardware-software co-design problem.** The T4 (Turing, 2018) Tensor Cores are built for FP16. The same INT8 config that hurts here would likely help on an A100 (Ampere, 2020), which has native INT8 hardware support and eliminates the dequantization step entirely. Next step: re-run on A100 to verify — if INT8 flips from slower to faster on Ampere, the rule becomes clear: match your precision to your hardware's compute primitives, not just your memory budget.

---

`Qwen2.5-0.5B · bitsandbytes · torch.profiler · Tesla T4 · Google Colab`
