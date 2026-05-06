# Migration to FlagGems C++ Wrapper Registration

## Overview

This migration replaces Python-level operator registration with C++ wrapper-based registration for significantly reduced CPU overhead.

## What Changed

### Before (Python-level registration)
- FlagGems operators were registered in Python using `torch.library.Library.impl()`
- Registration happened at Python import time in `torch_fl/__init__.py`
- Each operator call had higher CPU overhead due to Python dispatch layer

### After (C++ wrapper registration)
- FlagGems operators are registered in C++ using `TORCH_LIBRARY_IMPL`
- Registration happens at shared library load time (before Python runs)
- Direct C++ dispatch with minimal overhead

## Architecture

### New Files
- **`csrc/aten/FlagGemsRegistration.cpp`**: C++ operator registration using FlagGems C++ library
  - Registers ~80+ operators via `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`
  - Only compiled when FlagGems C++ library is found at build time

### Modified Files
- **`CMakeLists.txt`**: Added `find_package(FlagGems)` and conditional compilation
- **`csrc/CMakeLists.txt`**: Links against `FlagGems::operators` when available
- **`torch_fl/csrc/Module.cpp`**: Exposes `_flaggems_cpp_registration` flag to Python
- **`torch_fl/__init__.py`**: Detects C++ registration and skips Python fallback

## Build Requirements

### With C++ Wrapper (Recommended)
```bash
# 1. Build and install FlagGems with C++ extensions
cd /path/to/FlagGems
pip install -e . -v --config-settings=cmake.define.FLAGGEMS_BUILD_C_EXTENSIONS=ON

# 2. Build PyTorch-Plugin-FL (will auto-detect FlagGems)
cd /path/to/PyTorch-Plugin-FL
pip install -e . --no-build-isolation
```

### Without C++ Wrapper (Fallback)
If FlagGems C++ library is not found, the build will succeed with a warning and fall back to Python-level registration:
```
-- FlagGems C++ library not found. Operators will use Python-level registration (slower).
```

## Runtime Behavior

### C++ Registration Active
```python
import torch_fl
print(torch_fl._C._flaggems_cpp_registration)  # 1
# Operators registered in C++ at .so load time
# Python registration code is skipped
```

### Python Fallback
```python
import torch_fl
print(torch_fl._C._flaggems_cpp_registration)  # 0
# Falls back to Python-level registration
# Same functionality, higher CPU overhead
```

## Registered Operators

The following operators are registered via C++ wrapper when available:

### BLAS Operations
- `addmm`, `mm`, `bmm`

### Reductions
- `sum`, `sum.dim_IntList`
- `max`, `max.dim`, `max.dim_max`

### Normalization
- `rms_norm`, `fused_add_rms_norm`

### Tensor Operations
- `cat`, `embedding`, `embedding_backward`
- `argmax`, `nonzero`, `topk`

### Division Operations
- `div.*`, `divide.*`, `true_divide.*`, `floor_divide.*`
- `remainder.*`

### Other Operations
- `sort`, `sort.stable`
- `fill.*`, `softmax`, `softmax_backward`
- `zeros`, `exponential_`
- `copy_`, `to_copy`

### Custom FlagGems Operators
- `rotary_embedding`, `rotary_embedding_inplace`
- `reshape_and_cache_flash`, `flash_attn_varlen_func`
- `rwkv_mm_sparsity`, `rwkv_ka_fusion`

## Performance Impact

### Expected Improvements
- **Operator dispatch overhead**: ~30-50% reduction in CPU time for small tensor operations
- **Import time**: Slightly faster (C++ registration is more efficient)
- **Memory**: Minimal difference

### Benchmarking
```python
import torch
import torch_fl
import time

x = torch.randn(100, 100, device="flagos")
y = torch.randn(100, 100, device="flagos")

# Warmup
for _ in range(100):
    z = torch.mm(x, y)

# Benchmark
start = time.perf_counter()
for _ in range(10000):
    z = torch.mm(x, y)
torch_fl.flagos.synchronize()
elapsed = time.perf_counter() - start

print(f"Time per mm: {elapsed/10000*1e6:.2f} µs")
```

## Troubleshooting

### Build fails with "FlagGems not found"
This is just a warning. The build will succeed with Python fallback. To enable C++ registration:
1. Install FlagGems with C++ extensions: `pip install flag-gems --config-settings=cmake.define.FLAGGEMS_BUILD_C_EXTENSIONS=ON`
2. Ensure FlagGems installs CMake config files to a standard location
3. Set `CMAKE_PREFIX_PATH` if needed: `export CMAKE_PREFIX_PATH=/path/to/flaggems/install`

### Runtime error: "operator already registered"
This means both C++ and Python registration are active (shouldn't happen). Check:
```python
import torch_fl
print(torch_fl._USE_CPP_REGISTRATION)  # Should be True or False, not both
```

### Performance not improved
1. Verify C++ registration is active: `torch_fl._C._flaggems_cpp_registration == 1`
2. Check that FlagGems C++ library is properly linked: `ldd torch_fl/lib/libtorch_fl.so | grep operators`
3. Profile to ensure bottleneck is actually in dispatch (not kernel execution)

## Migration Checklist

- [x] Add FlagGems dependency to CMakeLists.txt
- [x] Create FlagGemsRegistration.cpp with TORCH_LIBRARY_IMPL
- [x] Update csrc/CMakeLists.txt to link FlagGems::operators
- [x] Add _flaggems_cpp_registration flag to Module.cpp
- [x] Update torch_fl/__init__.py to detect and skip Python registration
- [x] Maintain backward compatibility (Python fallback)
- [ ] Test with FlagGems C++ library installed
- [ ] Test without FlagGems C++ library (fallback mode)
- [ ] Benchmark performance improvements

## References

- FlagGems C++ wrapper documentation: `/nfs/hcr/repos/FlagGems/docs/add_a_cpp_wrapper.md`
- FlagGems C++ registration example: `/nfs/hcr/repos/FlagGems/src/flag_gems/csrc/cstub.cpp`
- PyTorch custom backend guide: https://pytorch.org/tutorials/advanced/extend_dispatcher.html
