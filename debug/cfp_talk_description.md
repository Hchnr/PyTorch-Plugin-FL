# PyTorch Conference — Call for Papers: Talk Description

---

## 中文版

智源联合合作伙伴共建的基于 FlagOS 多芯片统一后端的 PyTorch 设备插件（PyTorch-FL-Plugin）：

1）**跨芯片统一架构**：基于 PyTorch PrivateUse1 开放注册机制，结合 FlagGems 算子库（Triton 内核）与 FlagCX 通信库，通过可配置的 DispatchStub 后端分发架构，一套代码即可在 CUDA、NPU、MUSA 等不同芯片上运行大模型训练与推理；

2）**完全兼容 PyTorch**：不改变 PyTorch 原有使用方式，用户只需 `import torch_fl` 即可将模型运行在 `flagos` 设备上，零侵入、低成本接入，支持 DDP/FSDP 分布式训练与 AMP 混合精度。

3）**Out-of-tree 插件化架构**：采用 PyTorch 原生的 out-of-tree 后端扩展方式，不修改 PyTorch 主干代码，以独立插件形式提供多芯片支持。这一架构为各芯片厂商已有的 in-tree 定制分支提供了标准化的插件迁移路径，将分散的厂商 PyTorch fork 收敛为统一的插件接口，降低跨版本维护成本，同时保持与 PyTorch 上游的持续兼容；

4）**Stub 级分发与 Meta 复用**：区别于传统的逐算子注册方式，我们在 DispatchStub 层级接入后端实现，继承 PyTorch 原生的 structured kernel 体系，直接复用上游的 meta 函数（shape 推导、dtype 推断、内存布局计算等），仅替换计算内核。这使得每个算子的接入只需提供纯计算实现，无需重复编写元信息逻辑，大幅降低多后端适配成本。计算内核采用三级分发策略：优先使用 FlagGems 算子库（跨芯片统一的 Triton 内核），其次对接芯片厂商原生算子库，最后通过 CPU fallback 保证功能完整性。内置 fallback 追踪工具可按算子统计调用频次，指导内核迁移优先级。

---

## English Version

**A Unified Multi-Chip PyTorch Device Plugin Built on FlagOS**

BAAI and its partners present PyTorch-FL-Plugin, a PyTorch device plugin built on the FlagOS multi-chip backend that brings portable large model training and inference across heterogeneous accelerators:

1) **Unified cross-chip architecture**: Built on PyTorch's PrivateUse1 open registration mechanism, integrating the FlagGems operator library (Triton kernels) and the FlagCX communication library through a configurable DispatchStub backend architecture. A single codebase runs large model training and inference across CUDA, NPU, MUSA, and other accelerators;

2) **Fully compatible with PyTorch**: Requires zero modifications to existing PyTorch user code. A single `import torch_fl` makes the `flagos` device available, with out-of-the-box support for DDP/FSDP distributed training and AMP mixed precision — zero intrusion, low integration cost;

3) **Out-of-tree plugin architecture**: Built entirely as an out-of-tree backend extension with no patches to upstream PyTorch. This design provides a standardized migration path for chip vendors that currently maintain in-tree forks of PyTorch, consolidating fragmented vendor-specific branches into a unified plugin interface. The result is lower cross-version maintenance overhead and continuous compatibility with upstream PyTorch releases;

4) **Stub-level dispatch with meta reuse**: Rather than registering each operator end-to-end, we plug backend implementations in at the DispatchStub level and inherit PyTorch's structured kernel framework, directly reusing upstream meta functions for shape inference, dtype propagation, and memory layout computation. Only the compute kernel itself needs to be supplied per operator, eliminating redundant meta logic and significantly reducing the cost of multi-backend adaptation. Compute kernels follow a three-tier dispatch strategy: FlagGems (cross-chip Triton kernels) first, vendor-supplied native operator libraries second, and CPU fallback as the final safety net for functional completeness. A built-in fallback tracing tool profiles per-op call frequency to guide prioritized kernel migration.
