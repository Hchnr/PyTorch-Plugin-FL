import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_PATH = "/nfs/hcr/models/Qwen/Qwen3-0.6B"


class AtenOpCollector(TorchDispatchMode):
    def __init__(self):
        self.ops = defaultdict(int)  # op -> call count

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[str(func)] += 1
        return func(*args, **(kwargs or {}))


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
inputs = tokenizer("hello world", return_tensors="pt")

collector = AtenOpCollector()

# --- inference ---
with collector:
    with torch.no_grad():
        out = model(**inputs)

# --- training (forward + backward) ---
model.train()
with collector:
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()

# 输出结果
for op, count in sorted(collector.ops.items()):
    print(f"{op:60s} {count}")
