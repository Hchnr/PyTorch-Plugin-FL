"""
Collect a complete list of aten operators used during full inference and training.

Aligned with e2e profiling scripts:
  - Inference: model.generate() with greedy decoding (prefill + decode + sampling)
  - Training: forward + backward + AdamW optimizer step

Usage:
    python tests/manual/op_called_summary.py [--model PATH] [--tokens N] [--steps N]
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dummy_dataset import DummyTextDataset  # noqa: E402


class AtenOpCollector(TorchDispatchMode):
    """Collect ATen ops with call depth and count info."""

    def __init__(self):
        self.ops = defaultdict(int)  # op_name -> call count
        self.traces = []  # list of (depth, op_name)
        self._depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op_name = str(func)
        self.ops[op_name] += 1
        self.traces.append((self._depth, op_name))
        self._depth += 1
        try:
            return func(*args, **(kwargs or {}))
        finally:
            self._depth -= 1


def collect_inference_ops(model, tokenizer, max_new_tokens):
    """Run model.generate() (prefill + autoregressive decode + sampling)."""
    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language model.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Greedy decoding (same as e2e_qwen3_infer_cuda.py)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    collector = AtenOpCollector()
    with collector, torch.no_grad():
        output = model.generate(**gen_kwargs)

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    print(f"  Generated {new_tokens} tokens (greedy)")
    return collector


def collect_training_ops(model, tokenizer, args):
    """Run forward + backward + optimizer.step() (same as e2e_qwen3_train_cuda.py)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = model.device
    model.train()

    # Freeze unused parameters (same as e2e train script)
    dummy = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(input_ids=dummy, use_cache=False)
        out.logits.sum().backward()
    for _name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
        else:
            param.grad = None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=args.seq_len)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    collector = AtenOpCollector()
    data_iter = iter(dataloader)

    with collector:
        for step in range(args.steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"  Step {step + 1}/{args.steps} loss={loss.item():.4f}")

    return collector


def print_ops_summary(title, collector):
    """Print op call counts (sorted by name)."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")
    print(f"  {'op':<58s} {'count':>6}")
    print(f"  {'-' * 66}")
    for op, count in sorted(collector.ops.items()):
        print(f"  {op:<58s} {count:>6}")
    print(f"{'─' * 70}")
    print(f"  Total unique ops: {len(collector.ops)}")
    print(f"  Total calls:      {sum(collector.ops.values())}")


def print_depth_trace(title, collector, max_lines=200):
    """Print dispatch trace with depth (tree view)."""
    print(f"\n{'=' * 70}")
    print(f" {title} — Depth Trace")
    print(f"{'=' * 70}")
    print(f"  {'depth':<7} {'label':<12} {'op'}")
    print(f"  {'-' * 66}")

    traces = collector.traces
    n = len(traces)
    shown = min(n, max_lines)

    for idx in range(shown):
        depth, op = traces[idx]
        indent = "│ " * depth
        is_top = depth == 0
        next_depth = traces[idx + 1][0] if idx + 1 < n else 0
        is_leaf = next_depth <= depth
        if is_top and is_leaf:
            label = "[top+leaf]"
        elif is_top:
            label = "[top]"
        elif is_leaf:
            label = "[leaf]"
        else:
            label = "[mid]"
        print(f"  {depth:<7} {label:<12} {indent}{op}")

    if n > max_lines:
        print(f"  ... ({n - max_lines} more entries, use --trace-all to show)")

    max_depth = max(d for d, _ in traces) if traces else 0
    print(f"  {'-' * 66}")
    print(f"  Total entries: {n}, max depth: {max_depth}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect complete aten op list for inference and training"
    )
    parser.add_argument(
        "--model",
        default="/nfs/hcr/models/Qwen/Qwen3-0.6B",
        help="Path to model",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=64,
        help="Tokens to generate for inference",
    )
    parser.add_argument("--steps", type=int, default=3, help="Training steps")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Training seq length"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--trace-all",
        action="store_true",
        help="Print full depth trace (no truncation)",
    )
    args = parser.parse_args()

    device = "cuda"
    max_trace_lines = None if args.trace_all else 200

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print()

    # Load model (fp16 for inference, same as e2e scripts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    # Force eager attention to match e2e profiling scripts
    model.model.layers[0].self_attn.config._attn_implementation = "eager"

    # --- Inference: generate (prefill + decode + sampling) ---
    print("=== Collecting inference ops (generate) ===")
    infer_collector = collect_inference_ops(model, tokenizer, args.tokens)

    # --- Training: forward + backward + AdamW step ---
    # Reload in fp32 for training (same as e2e_qwen3_train_cuda.py)
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)

    print("\n=== Collecting training ops (forward + backward + AdamW) ===")
    train_collector = collect_training_ops(model, tokenizer, args)

    # --- Reports ---
    print_ops_summary(
        "INFERENCE OPS (generate: prefill + decode)", infer_collector
    )
    print_ops_summary(
        "TRAINING OPS (forward + backward + AdamW step)", train_collector
    )

    # Depth traces
    kw = {} if max_trace_lines is None else {"max_lines": max_trace_lines}
    print_depth_trace("INFERENCE", infer_collector, **kw)
    print_depth_trace("TRAINING", train_collector, **kw)

    # Incremental analysis
    infer_ops = set(infer_collector.ops.keys())
    train_ops = set(train_collector.ops.keys())
    train_only = train_ops - infer_ops
    infer_only = infer_ops - train_ops
    all_ops = infer_ops | train_ops

    print(f"\n{'=' * 70}")
    print(" TRAINING-ONLY OPS (backward + optimizer, not in inference)")
    print(f"{'=' * 70}")
    for op in sorted(train_only):
        print(f"  {op:60s} {train_collector.ops[op]}")
    print(f"{'─' * 70}")
    print(f"  Count: {len(train_only)}")

    print(f"\n{'=' * 70}")
    print(" INFERENCE-ONLY OPS (generate/sampling, not in training)")
    print(f"{'=' * 70}")
    for op in sorted(infer_only):
        print(f"  {op:60s} {infer_collector.ops[op]}")
    print(f"{'─' * 70}")
    print(f"  Count: {len(infer_only)}")

    print(f"\n{'=' * 70}")
    print(" SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Inference unique ops: {len(infer_ops)}")
    print(f"  Training unique ops:  {len(train_ops)}")
    print(f"  Combined unique ops:  {len(all_ops)}")
    print(f"  Training-only ops:    {len(train_only)}")
    print(f"  Inference-only ops:   {len(infer_only)}")


if __name__ == "__main__":
    main()
