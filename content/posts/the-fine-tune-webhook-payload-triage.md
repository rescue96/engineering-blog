---
title: "The fine-tune: teaching an 8B model to triage webhook failures in 38ms"
date: 2026-05-14
author: "Tomás Rey"
tags: ["ai", "fine-tuning", "inference", "reliability"]
---

When you deliver nine million webhooks a day across 14,000 distinct customer endpoints, failure is not an anomaly; it is continuous background noise. But the hardest part is rarely the failure itself — it is explaining to a customer why their destination choked.

Downstream endpoints almost never fail with clean HTTP semantics. A merchant's custom checkout proxy returns `200 OK` with `{"error": "signature_expired"}` buried in the JSON payload. A legacy enterprise ERP terminates the TCP socket after receiving the headers. A misconfigured WAF returns an HTML page complete with JavaScript challenge scripts instead of JSON.

For five months, we routed unclassified delivery failures through an external frontier LLM to parse response bodies, classify root causes into clean categories (such as signature mismatch, payload truncation, or schema drift), and draft remediation hints in the customer dashboard. It worked, but it was an operational disaster: 1,400ms p95 latency, $4,800 every month in token bills, and sensitive customer error payloads leaving our VPC.

This post is the story of how we replaced that frontier API with an in-house fine-tuned 8B model: why 2,400 hand-curated examples beat 50,000 synthetic ones, how constrained decoding eliminated schema drift, and what the numbers look like in production.

## Prompts don't belong in the data path

The prototype was three hundred lines of TypeScript wrapping a hosted model API. Whenever a delivery failed after three attempts with an ambiguous body, a worker bundled the HTTP status, headers, and the first 2KB of the response body into a prompt, asked for structured JSON, and parsed the response.

It delivered what marketing called "AI-powered diagnostics." But as delivery volume doubled, the flaws compounded:

1. **Latency spikes:** The hosted API averaged 900ms and regularly stretched past 2,500ms during provider incidents. When you are processing 320,000 failure events a day, queuing those jobs backed up our worker pools.
2. **Schema fragility:** Even with JSON mode enabled, roughly one in every 250 responses hallucinated a new category key (e.g. `RATE_LIMITED_BY_DESTINATION` instead of `RATE_LIMIT_DESTINATION`), requiring defensive regex fallbacks.
3. **Data sovereignty:** Several enterprise customers flagged that webhook response fragments containing partial customer records were crossing third-party LLM inference boundaries.

The frontier model was doing too much thinking. We did not need an agent capable of writing poetry or reasoning about quantum physics; we needed a specialized classifier that understood HTTP edge cases, truncated stack traces, and our exact taxonomy.

## Synthetic data was a trap

Our first instinct was to generate our way out of the data shortage. We wrote a generator script that used a frontier model to invent 50,000 synthetic webhook failure bodies across our eight failure categories, split them 80/20, and fine-tuned an open-weights 8B model using LoRA.

On the synthetic validation set, the model was brilliant: 98.2% accuracy.

When we mirrored shadow production traffic to it the following Monday, accuracy collapsed to 61.4%.

Synthetic data was a trap because LLMs generate *clean* failures. A synthetic Django stack trace looks like textbook Python; a synthetic rate-limit response has a polite message and clean headers. Real production failures are disgusting. They contain truncated UTF-8 sequences from unbuffered proxies, half-emitted XML inside JSON strings, PHP warnings injected before JSON headers, and Cloudflare HTML 520 pages.

We discarded the synthetic dataset entirely and rebuilt from scratch:

```sql
-- Extract real, high-ambiguity failures for human labeling
SELECT destination_id,
       status_code,
       substr(response_body, 1, 1500) AS sample_body,
       count(*)                        AS frequency
FROM delivery_failures
WHERE created_at > now() - INTERVAL '14 days'
  AND status_code NOT IN (404, 410)
GROUP BY destination_id, status_code, substr(response_body, 1, 1500)
ORDER BY frequency DESC
LIMIT 3000;
```

Two platform engineers spent three days labeling 2,400 genuine failure snippets, annotating edge cases like silent 200s and WAF challenges. That dataset was twenty times smaller than our synthetic set, but validation accuracy on live production traffic leaped from 61.4% to 96.8%. Quality of error distribution trumped quantity of synthetic perfection.

## The recipe: LoRA with masked prompt loss

We selected a base open-weights 8B instruction model and fine-tuned it using Low-Rank Adaptation (LoRA). The key realization during training was loss masking: if you calculate cross-entropy loss over the entire token sequence, the model wastes parameter capacity learning to predict the repetitive structure of HTTP headers and boilerplate prompt tokens.

We masked the prompt tokens so gradients only propagated from the classification response:

```python
# training/triage_lora.py
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_model_and_lora(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, lora_config)
```

The training ran on two rented A100 GPUs for four hours, costing under $25 in compute. The adapter weights were just 160MB.

## Constrained decoding: zero schema drift

Fine-tuning taught the model our classification taxonomy, but autoregressive generation still had a non-zero chance of generating a stray token or trailing comma that would break JSON parsing.

Instead of post-processing or retry loops, we deployed the model on an internal inference engine using grammar-constrained decoding. The sampler restricts valid next tokens at each step to characters allowed by our Pydantic schema:

```python
# inference/schema.py
from pydantic import BaseModel
from typing import Literal

class TriageResult(BaseModel):
    category: Literal[
        "AUTH_SIGNATURE_MISMATCH",
        "SCHEMA_VALIDATION_ERROR",
        "RATE_LIMIT_WAF",
        "RATE_LIMIT_APPLICATION",
        "PAYLOAD_TRUNCATION",
        "ENDPOINT_TIMEOUT",
        "SILENT_SUCCESS_ERROR",
        "UPSTREAM_INTERNAL_ERROR",
    ]
    is_transient: bool
    confidence_score: float
    remediation_hint: str
```

Under constrained decoding, the probability of an invalid JSON output is zero by construction. The engine physically cannot sample a token that violates the state machine of the grammar.

## The numbers in production

We deployed the fine-tuned adapter to two internal inference nodes powered by NVIDIA L4 GPUs running vLLM with continuous batching. We ran the old hosted API and the new in-house model side-by-side for three weeks on 100% of failed deliveries:

```text
Metric                 Hosted Frontier API     Fine-Tuned 8B (vLLM)
--------------------------------------------------------------------
p50 Latency            910 ms                  19 ms
p99 Latency            2,420 ms                38 ms
Accuracy (Live Eval)   95.2%                   97.4%
Invalid JSON Rate      0.38%                   0.00%
Monthly Compute Cost   ~$4,800                 ~$165
Data Leaves VPC        Yes                     No
```

The fine-tuned model was not only 60 times faster; it was actually *more* accurate than the generalist frontier model. The frontier model constantly stumbled on custom enterprise headers and silent `200 OK` errors with `"status": "FAILED"` bodies. The 8B model had seen hundreds of them during training and classified them effortlessly.

## What we would do differently

Two honest lessons from the process:

First, beware of spurious prompt correlations. In our first training run, we included the HTTP `Date` header in the input prompt. The model quickly learned that failures clustered around 00:00 UTC were almost always rate-limit resets, and it began assigning high rate-limit probabilities to completely unrelated database timeouts that occurred at midnight. Stripping temporal headers restored unbiased semantic classification.

Second, don't over-quantize without checking tail distributions. We tested 4-bit AWQ quantization to see if we could fit the model onto an older, cheaper T4 GPU. Overall accuracy dropped by only 1.2%, which looked acceptable on paper — until we broke down the metrics by class. The accuracy for rare, high-severity errors (`SILENT_SUCCESS_ERROR`) dropped by 18%. We stayed on FP8 on modern L4s.

Frontier models are incredible exploratory engines: they tell you what is possible and help you discover the taxonomy your system needs. But once you know what you are measuring, pragmatic [AI implementation and AI fine-tuning](https://teckgeekz.com/ai) for dedicated models gives you what actually belongs in a production pipeline: predictable latency, zero data leakage, and a bill that does not make finance page on-call.
