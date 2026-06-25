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

