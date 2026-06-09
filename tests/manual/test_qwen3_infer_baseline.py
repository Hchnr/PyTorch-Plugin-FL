"""
Qwen3 Inference Baseline Test (native PyTorch + CUDA, no torch_fl)

Mirrors tests/integration/test_qwen3_infer.py but runs on CUDA with
eager attention to provide a fair performance baseline.

Usage:
    python tests/manual/test_qwen3_infer_baseline.py [--model PATH] [--max-new-tokens N] [--runs N]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 inference baseline (native CUDA)")
    parser.add_argument(
        "--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B", help="Path to Qwen3 model"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--runs", type=int, default=4, help="Number of inference runs")
    parser.add_argument(
        "--device", default="cuda:0", help="Device to run on (default: cuda:0)"
    )
    return parser.parse_args()


def run_inference(model, inputs, max_new_tokens, device):
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    return output, elapsed, new_tokens


def main():
    args = parse_args()
    device = args.device

    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model: {args.model}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Attention: eager")
    print()

    # Load model — match the integration test config exactly
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    # Force eager attention to match the integration test
    model.model.layers[0].self_attn.config._attn_implementation = "eager"
    print(f"Model loaded in {time.time() - t0:.2f}s")
    print(f"Model device: {next(model.parameters()).device}")
    print()

    # Prepare input — same prompt as integration test
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
    inputs = tokenizer([text], return_tensors="pt").to(device)
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print()

    # Run inference multiple times
    labels = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"]
    for i in range(args.runs):
        label = labels[i] if i < len(labels) else f"Run {i + 1}"
        output, elapsed, new_tokens = run_inference(
            model, inputs, args.max_new_tokens, device
        )
        tps = new_tokens / elapsed if elapsed > 0 else 0
        print(f"  {label}: {elapsed:.2f}s, {new_tokens} tokens, {tps:.2f} tok/s")

    # Decode last output
    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"\nGenerated output:\n{decoded}")


if __name__ == "__main__":
    main()
