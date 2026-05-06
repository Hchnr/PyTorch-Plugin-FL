# Plan: mm 算子跨后端统一 C++ Wrapper

## Context

当前 PyTorch-Plugin-FL 通过 Python 层（`integration.py`）将 FlagGems 的 Triton kernel 注册到 `PrivateUse1` dispatch key。目标是为 `mm` 算子实现一个 C++ 层的统一 wrapper，将 meta（shape 推断）和 impl（kernel 调用）分离，并支持 CUDA 和 FlagGems 两个后端的分发。

## 架构设计

### 核心思路：structured kernel 模式

PyTorch 原生 CUDA 的 `mm` 使用 **structured kernel** 模式（见 `RegisterCUDA_0.cpp:16047`）：
- `op.meta(self, mat2)` — 调用 `at::meta::structured_mm::meta()`，做 shape check 和 output tensor 分配
- `op.impl(self, mat2, out)` — 调用后端具体实现（`TORCH_IMPL_FUNC(mm_out_cuda)`）

我们的统一 wrapper 复用这套模式，但 impl 部分根据运行时后端分发到 CUDA 或 FlagGems。

### 分发策略

统一 wrapper 注册在 `PrivateUse1` dispatch key 上。在 `impl` 阶段，通过检查 tensor 的实际设备类型（`self.device().type()`）决定调用哪个后端：
- `kCUDA` → 调用 `at::native::mm_out_cuda`（或等价的 `at::mm_out` 走 CUDA dispatch）
- `kPrivateUse1`（flagos 设备）→ 调用 `flag_gems::mm_out_tensor`

**注意**：flagos 设备上的 tensor 实际内存在 CUDA 上，但 dispatch key 是 PrivateUse1。FlagGems 的 `mm_out_tensor` 直接接受这类 tensor（它只关心 data_ptr 和 CUDA stream，不检查 device type）。

## 文件变更

### 新建：`csrc/aten/native/mm.h`
声明统一 wrapper 的 meta 和 impl 函数：
```cpp
namespace at::native::flagos {
  at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);
  at::Tensor& mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out);
}
```

### 新建：`csrc/aten/native/mm.cpp`
实现 meta + impl 分离的统一 wrapper：

```cpp
#include "mm.h"
#include <flag_gems/operators.h>
#include <ATen/ops/mm_native.h>   // at::native::mm_out (CUDA path via redispatch)

namespace at::native::flagos {

// meta: shape 推断，复用 PyTorch 原生逻辑
static void mm_meta(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(self.size(1) == mat2.size(0),
    "mat1 and mat2 shapes cannot be multiplied (",
    self.size(0), "x", self.size(1), " and ",
    mat2.size(0), "x", mat2.size(1), ")");
  // resize output
  if (out.sizes() != at::IntArrayRef({self.size(0), mat2.size(1)})) {
    out.resize_({self.size(0), mat2.size(1)});
  }
}

// impl: 后端分发
static void mm_impl(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  // flagos 设备上的 tensor 实际在 CUDA 内存，FlagGems 直接处理
  flag_gems::mm_out_tensor(self, mat2, out);
  // 若将来需要支持纯 CUDA tensor（非 flagos），可在此加分支：
  // at::cuda::mm_out(out, self, mat2);
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty({self.size(0), mat2.size(1)}, self.options());
  mm_meta(self, mat2, out);
  mm_impl(self, mat2, out);
  return out;
}

at::Tensor& mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  mm_meta(self, mat2, out);
  mm_impl(self, mat2, out);
  return out;
}

} // namespace at::native::flagos
```

### 修改：`csrc/aten/FlagosMinimal.cpp`
在 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` 块中添加 mm 注册：

```cpp
#include "native/mm.h"

// 在 namespace at::flagos 的匿名 namespace 中添加：
at::Tensor wrapper_mm(const at::Tensor& self, const at::Tensor& mat2) {
  return at::native::flagos::mm(self, mat2);
}
at::Tensor& wrapper_mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  return at::native::flagos::mm_out(self, mat2, out);
}

// 在 TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) 中添加：
m.impl("mm", wrapper_mm);
m.impl("mm.out", wrapper_mm_out);
```

### 修改：`csrc/CMakeLists.txt`
确保链接 `FlagGems::operators`（已有 `MIGRATION_CPP_WRAPPER.md` 描述的条件链接逻辑，需确认 mm.cpp 被 glob 到）。当前 `file(GLOB_RECURSE SOURCE_FILES ...)` 会自动包含新文件，无需额外修改。

但需要确认 `FlagGems::operators` 已链接：
```cmake
# 条件链接（已在 MIGRATION_CPP_WRAPPER.md 中规划）
if(TARGET FlagGems::operators)
  target_link_libraries(${LIBRARY_NAME} PRIVATE FlagGems::operators)
endif()
```

## 关键依赖

- `flag_gems::mm_out_tensor` — 声明在 `/nfs/hcr/repos/FlagGems/include/flag_gems/operators.h:46`，实现在 `/nfs/hcr/repos/FlagGems/lib/mm.cpp:264`
- `at::meta::structured_mm` — PyTorch 原生 meta 类，`TORCH_META_FUNC(mm)` 定义在 `pytorch/aten/src/ATen/native/LinearAlgebra.cpp:202`
- 现有注册模式参考：`csrc/aten/FlagosMinimal.cpp`（PrivateUse1 注册）和 `csrc/aten/native/Minimal.cpp`（native 实现）

## 关于 meta/impl 抽象

FlagGems 的 `mm.cpp` 目前没有分离 meta 和 impl（`mm_tensor` 内部直接 `at::empty` + kernel 调用）。我们在 `at::native::flagos::mm_meta` 中独立实现 shape 推断（逻辑与 PyTorch `TORCH_META_FUNC(mm)` 一致），在 `mm_impl` 中调用 FlagGems 的 `mm_out_tensor`（它接受预分配的 out tensor，正好对应 impl 语义）。

## 验证方式

```python
import torch
import torch_fl

x = torch.randn(128, 256, device="flagos")
y = torch.randn(256, 64, device="flagos")

# 基本正确性
z = torch.mm(x, y)
assert z.shape == (128, 64)

# out 变体
out = torch.empty(128, 64, device="flagos")
torch.mm(x, y, out=out)
assert out.shape == (128, 64)

# 与 CPU 结果对比
x_cpu = x.cpu()
y_cpu = y.cpu()
z_cpu = torch.mm(x_cpu, y_cpu)
assert torch.allclose(z.cpu(), z_cpu, atol=1e-3)
```

同时确认 Python 层的 `integration.py` 中 `mm` 不再重复注册（或注册被 C++ 层覆盖不报错）。
