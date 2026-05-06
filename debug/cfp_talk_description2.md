# PyTorch Conference — Call for Papers: Talk Description (Condensed)

---

## 中文版

智源联合合作伙伴共建的基于 FlagOS 多芯片统一后端的 PyTorch 设备插件（PyTorch-FL-Plugin）：

1）**跨芯片统一架构**：基于 PyTorch PrivateUse1 开放注册机制，结合 FlagGems 算子库与 FlagCX 通信库，一套代码即可在 CUDA、NPU、MUSA 等异构加速器上运行大模型训练与推理，完全兼容 PyTorch 原有使用方式与 DDP/FSDP 分布式训练；

2）**Out-of-tree 插件化架构**：以独立插件形式提供多芯片支持，不修改 PyTorch 主干代码，为各芯片厂商的 in-tree 定制分支提供标准化的插件迁移路径，将分散的厂商 fork 收敛为统一接口；

3）**最小化算子适配成本**：复用 PyTorch 原生的算子框架逻辑，每个算子仅需提供纯计算内核即可接入。计算内核采用三级分发：FlagGems Triton 内核优先，芯片厂商原生算子库次之，CPU fallback 兜底，并内置追踪工具指导内核迁移优先级。

---

## English Version

**A Unified Multi-Chip PyTorch Device Plugin Built on FlagOS**

BAAI and its partners present PyTorch-FL-Plugin, a PyTorch device plugin built on the FlagOS multi-chip backend for portable large model training and inference across heterogeneous accelerators:

1) **Unified cross-chip architecture**: Built on PyTorch's PrivateUse1 open registration mechanism with the FlagGems operator library and FlagCX communication library, enabling a single codebase to run across CUDA, NPU, MUSA, and other accelerators with full compatibility with native PyTorch workflows including DDP/FSDP distributed training;

2) **Out-of-tree plugin architecture**: Delivers multi-chip support as a standalone plugin with zero patches to upstream PyTorch, providing a standardized migration path that consolidates fragmented vendor-specific in-tree forks into a unified plugin interface;

3) **Minimal per-operator adaptation cost**: Reuses PyTorch's native operator framework logic so that each operator only requires a bare compute kernel to integrate. Kernels follow a three-tier dispatch: FlagGems Triton kernels first, vendor-native operator libraries second, CPU fallback as the safety net, with built-in tracing to guide prioritized kernel migration.

---

## Benefits to the Ecosystem / 对生态的价值

### 中文版

1）**降低芯片厂商接入 PyTorch 的门槛**：厂商无需维护独立的 PyTorch fork，只需按统一接口提供计算内核即可接入，大幅减少跨版本适配与长期维护成本；

2）**促进 PyTorch 多芯片生态的标准化**：为当前碎片化的厂商后端实现提供一条收敛路径，推动社区形成统一的 out-of-tree 插件规范，减少生态重复建设；

3）**让用户在异构硬件间无缝迁移**：模型代码与训练脚本无需任何修改即可跨芯片运行，降低用户对单一硬件平台的锁定风险；

4）**为开源算子库提供落地通道**：FlagGems 等跨芯片 Triton 算子库可通过本插件直接服务于真实的训练与推理场景，形成"算子库开发—插件集成—模型验证"的闭环。

### English Version

1) **Lowers the barrier for chip vendors to integrate with PyTorch**: Vendors no longer need to maintain standalone PyTorch forks — they only need to supply compute kernels against a unified interface, significantly reducing cross-version porting and long-term maintenance costs;

2) **Drives standardization of PyTorch's multi-chip ecosystem**: Offers a convergence path for today's fragmented vendor backend implementations, encouraging the community to adopt a common out-of-tree plugin convention and reducing duplicated ecosystem effort;

3) **Enables seamless model portability across heterogeneous hardware**: Model code and training scripts run across different accelerators without modification, reducing user lock-in to any single hardware platform;

4) **Provides a production pathway for open-source operator libraries**: Cross-chip Triton operator libraries such as FlagGems gain a direct route into real-world training and inference workloads through this plugin, closing the loop from operator development to model-level validation.



Researcher, AI Framework Development Group, Beijing Academy of Artificial Intelligence (BAAI)