
A PyTorch plugin built on FlagOS to provide a unified and efficient multi-chip support.

## Overview

This package registers FlagGems' high-performance Triton kernels as a custom PyTorch device backend using PyTorch's PrivateUse1 extension mechanism. When you import `torch_fl`, all FlagGems operators are automatically registered for the `flagos` device.

## Installation

### Prerequisites

- Python == 3.12
- PyTorch == 2.9.0
- FlagGems == 4.2.1

### Build from source

```bash
cd PyTorch-Plugin-FL
# For CUDA platform
pip install -e . --no-build-isolation

# For MACA platform
ACCELERATOR=maca pip install -e . --no-build-isolation
```

## Usage

### Import Order (MACA Platform Only)

**IMPORTANT**: On MetaX (MACA) hardware, you must import `torch_fl` **before** importing `torch`:

```python
import torch_fl  # Must come first on MACA
import torch

# Now safe to use torch and torch_fl
x = torch.randn(100, 100, device="flagos")
```

**Why?** PyTorch's bundled CUDA 12.x runtime is ABI-incompatible with MACA's cu-bridge (CUDA 11.6 compatibility layer). `torch_fl` loads a shim library that provides the required symbol versions before PyTorch's `.so` files are loaded. If you import `torch` first, the incompatible symbols are already loaded and the shim cannot fix them.

On CUDA platforms, import order doesn't matter.

### Basic Usage

```python
import torch
import torch_fl  # Automatically registers FlagGems operators for "flagos" device

# Check device and FlagGems status
print(f"Device available: {torch_fl.flagos.is_available()}")
print(f"FlagGems enabled: {torch_fl.is_flaggems_enabled()}")
print(f"Registered ops: {len(torch_fl.get_registered_ops())}")

# Create tensors on flagos device - operations automatically use FlagGems kernels
x = torch.randn(1000, 1000, device="flagos")
y = torch.randn(1000, 1000, device="flagos")

# All operations use FlagGems kernels (no flag_gems.enable() needed!)
z = x + y
mm_result = torch.mm(x, y)
softmax_result = torch.softmax(x, dim=-1)

# Move tensors between devices
cpu_tensor = torch.randn(3, 3)
flagos_tensor = cpu_tensor.to("flagos")
back_to_cpu = flagos_tensor.cpu()

# Use device context manager
with torch_fl.flagos.device(0):
    a = torch.randn(10, 10, device="flagos")


# Check if FlagGems is available
torch_fl.is_flaggems_available()  # -> bool

# Check if FlagGems operators are registered
torch_fl.is_flaggems_enabled()  # -> bool

# Get list of registered operator names
torch_fl.get_registered_ops()  # -> List[str]

# Device module (torch_fl.flagos)
torch_fl.flagos.is_available()  # -> bool
torch_fl.flagos.device_count()  # -> int
torch_fl.flagos.current_device()  # -> int
torch_fl.flagos.set_device(device_id)
```

## Project Structure

```
PyTorch-Plugin-FL/
├── CMakeLists.txt                  # Top-level CMake build
├── setup.py / pyproject.toml       # Python packaging
├── csrc/                           # C++ source code
│   ├── aten/                       # Operator implementations & registration
│   │   ├── FlagosMinimal.cpp       #   PrivateUse1 operator dispatch registrations
│   │   └── native/                 #   Native implementations
│   │       ├── Minimal.cpp         #     Basic tensor ops (empty, set_, copy, clone, etc.)
│   │       └── Common.h
│   └── runtime/                    # Device runtime layer
│       ├── FlagosFunctions.cpp/h   #   Device management (set/get/exchange device)
│       ├── FlagosDeviceAllocator.* #   GPU memory allocation (wraps cudaMalloc)
│       ├── FlagosHostAllocator.*   #   Pinned host memory allocation
│       ├── FlagosGenerator.*       #   Random number generator
│       ├── FlagosHooks.*           #   PyTorch backend hooks integration
│       ├── FlagosGuard.*           #   Device guard (RAII device switching)
│       └── FlagosException.h       #   Error handling
├── accelerator/
│   ├── include/flagos.h            # C API (foSetDevice, foMalloc, foStream, etc.)
│   ├── csrc/
│   │   ├── cuda/                   # CUDA/MACA runtime SDK implementation
│   │   └── maca/                   # MACA-specific C shim (cudart_shim.c, libcudart.version)
│   └── maca/                       # MACA Python compatibility layer
│       ├── _maca_compat.py         #   torch.cuda patches for MetaX hardware
│       └── _maca_cudart_shim.py    #   cudart shim build & load
├── torch_fl/                       # Python package
│   ├── __init__.py                 #   Entry point: registers FlagGems ops on import
│   ├── integration.py              #   FlagGems operator registration logic
│   ├── distributed.py              #   Distributed training (DDP/FSDP support for flagos)
│   ├── _utils.py                   #   Utility functions
│   ├── flagos/                     #   Device module (torch_fl.flagos)
│   │   ├── __init__.py             #     Device APIs (set_device, synchronize, etc.)
│   │   ├── random.py               #     RNG APIs
│   │   └── meta.py                 #     Device metadata
│   ├── csrc/                       #   Python C binding
│   │   └── Module.cpp
│   └── lib/                        #   Built shared libraries
└── tests/                          # Test suite
    ├── common/                     #   Shared utilities
    │   └── dummy_dataset.py        #     Synthetic dataset for training tests
    ├── integration/                #   CI integration tests (cuda + flagos, same scripts)
    │   ├── test_ops.py             #     Basic ops and tensor tests (pytest, --device cuda|flagos)
    │   ├── test_qwen3_infer.py     #     Qwen3 end-to-end inference (--device cuda|flagos)
    │   └── test_qwen3_train.py     #     Qwen3 end-to-end training (--device cuda|flagos, single/DDP/FSDP)
    └── manual/                     #   Manual scripts for interactive exploration
```

## Testing

The test suite is split into integration tests (CI) and manual scripts (interactive exploration).

```
tests/
├── common/              # Shared utilities (DummyTextDataset)
├── integration/         # CI integration tests — same scripts for all platforms
└── manual/              # Manual scripts for interactive exploration
```

### Integration tests (`tests/integration/`)

All three scripts accept a `--device` flag (`cuda` or `flagos`) and are otherwise identical — this is the core goal of `torch_fl`.

**Basic ops and tensor tests (pytest):**

```bash
# Disable auto-loading of FlagCX
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

pytest tests/integration/test_ops.py -v --device cuda    # CUDA
pytest tests/integration/test_ops.py -v --device flagos  # MACA
```

**Qwen3 inference (pytest):**

```bash
pytest tests/integration/test_qwen3_infer.py -v -s --device cuda
pytest tests/integration/test_qwen3_infer.py -v -s --device flagos
pytest tests/integration/test_qwen3_infer.py -v -s --device cuda --model /path/to/Qwen3-0.6B --max-new-tokens 64
```

**Qwen3 training (pytest, single GPU):**

```bash
pytest tests/integration/test_qwen3_train.py -v -s --device cuda  --steps 10
pytest tests/integration/test_qwen3_train.py -v -s --device flagos --steps 10
```

## MACA Specific
Add the MACA cu-bridge library path to `LD_LIBRARY_PATH`, for example:

```bash
export LD_LIBRARY_PATH=/opt/maca-3.3.0/tools/cu-bridge/lib:$LD_LIBRARY_PATH
```

Install Triton (versions for MACA).

Install PyTorch-Plugin-FL.
```bash
ACCELERATOR=maca pip install -e . --no-build-isolation -v
```
