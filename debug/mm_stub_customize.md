# 重构 mm 算子 stub 分发机制

## Context

当前 `mm_stub.h` / `mm_stub.cpp` 使用了 PyTorch 的 `DispatchStub` 宏体系（`DECLARE_DISPATCH`, `DEFINE_DISPATCH`, `REGISTER_PRIVATEUSE1_DISPATCH`, `REGISTER_NO_CPU_DISPATCH`），以及 `structured_mm_out_cuda` 类。这套机制对我们来说过重——我们的 stub 只需根据 `get_backend_for_op()` 返回的 `FlagosDevice` 做一个 switch 分发。

## 改动方案

### 1. 删除 `mm_stub.h` 和 `mm_stub.cpp`

这两个文件完全依赖 PyTorch 的 DispatchStub 宏，重构后不再需要。

### 2. 修改 `mm.cpp` — 将分发逻辑内联到 `impl()`

把原来 `mm_stub.cpp` 中的 backend switch 逻辑直接放到 `structured_mm_out_flagos::impl()` 中：

```cpp
void structured_mm_out_flagos::impl(const Tensor& self, const Tensor& mat2) {
  auto backend = flagos::get_backend_for_op("mm");
  flagos::log_dispatch("mm", backend);
  switch (backend) {
    case flagos::FlagosDevice::FlagOS:
      flag_gems::mm_out_tensor(self, mat2, out_);
      break;
    case flagos::FlagosDevice::CUDA:
      // 直接调用 PyTorch 原生 CUDA structured op
      at::native::structured_mm_out_cuda::impl(self, mat2, out_);
      break;
    case flagos::FlagosDevice::NPU:
    case flagos::FlagosDevice::MUSA:
      TORCH_CHECK(false, "mm: NPU/MUSA backend not yet implemented");
      break;
  }
}
```

移除 `#include "mm_stub.h"` 和 `#include <ATen/native/DispatchStub.h>`，新增 `#include <flag_gems/operators.h>` 和 `#include <ATen/ops/mm_native.h>`（CUDA 路径需要）。

### 3. `mm.h` 保持不变

`structured_mm_out_flagos` 继承 `at::meta::structured_mm` 提供 shape 校验（`meta()`），这是有用的，不属于 stub 层面的复杂度。

### 4. `Register.cpp` — 移除 DispatchStub 相关 include

删除 `#include <ATen/native/DispatchStub.h>`（不再需要）。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `csrc/aten/functional_ops/mm_stub.h` | 删除 |
| `csrc/aten/functional_ops/mm_stub.cpp` | 删除 |
| `csrc/aten/functional_ops/mm.cpp` | 修改：内联分发逻辑 |
| `csrc/aten/Register.cpp` | 修改：移除 DispatchStub include |

## 验证

1. 编译通过（`python setup.py build` 或项目构建命令）
2. 运行 `tests/manual/op_called_summary.py` 确认 mm 算子正常分发
