# Model-right-sizing

# Model Right-Sizing for Inference: DistilBERT vs Quantized DistilBERT

## Common Assumption

A common assumption in ML deployment is:

> Smaller or compressed models are always faster and better for inference.

For example, if a model has fewer parameters, lower precision, or smaller storage size, we may assume it will automatically reduce latency and improve deployment efficiency.

However, real inference performance is not always determined only by model size or parameter count. Latency and throughput can also depend on hardware, runtime support, operator implementation, batching, memory access, and execution overhead.

## Issue

The main issue is that model selection is often based on paper-level metrics such as:

- Parameter count
- Model size
- FLOPs
- Accuracy

But these metrics do not always explain real deployment behavior.

A compressed model may be smaller, but it may not always be significantly faster. Similarly, a model with fewer parameters may still have latency bottlenecks depending on how inference is executed.

This makes model right-sizing an important ML systems problem.

## Task

In this experiment, I compare:

1. Original DistilBERT model
2. Dynamically quantized DistilBERT model

Both models are evaluated on the same SST-2 sentiment classification validation dataset.

I measure:

- Accuracy
- Model size
- Inference latency per sample
- Throughput
- Memory / deployment efficiency

## Goal

The goal is to understand whether quantization actually improves deployment-relevant metrics, not just model size.

More specifically, this experiment asks:

> When should we choose a quantized model over the original model for inference deployment?

## Hypothesis

I expect dynamic quantization to reduce model size and memory usage while preserving similar accuracy.

However, I do not assume that latency will improve in the same proportion as model size reduction, because real inference speed depends on runtime behavior, hardware support, and execution overhead.

## Expected Impact / Result

This experiment creates a small, reproducible benchmarking workflow for model right-sizing.

The expected result is a decision framework that helps compare original and quantized models using both model-level and deployment-level metrics.

Instead of choosing a model only because it is smaller, this experiment helps determine whether the compressed model is actually better under practical constraints such as latency, throughput, accuracy, and storage cost.

## Why This Matters

In real AI deployment, the “best” model is not always the largest or the smallest model.

The right model depends on the deployment constraint:

- If accuracy matters most, the original model may be preferred.
- If storage or memory is limited, the quantized model may be better.
- If latency improves without much accuracy loss, quantization may be useful.
- If latency does not improve significantly, model size alone is not enough to justify compression.

This is why benchmarking actual inference behavior is important for model right-sizing.

# NEW 
# When Quantization Does Not Speed Up LLMs

This is an experiment to understand **whether quantization actually improves LLM inference speed on real hardware**, or whether it mainly reduces memory while introducing runtime overhead.

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

