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
