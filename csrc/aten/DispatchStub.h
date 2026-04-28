// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "Common.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace at::native::flagos {

// Lightweight dispatch stub replacing PyTorch's DispatchStub.
//
// A stub owns a stub_name (e.g. "mm") and a set of per-backend kernel
// pointers. Multiple ops can share one stub (e.g. "mm" and "mm.out"
// share the same kernels). The op name used for config lookup and
// logging is passed at the call site via dispatch_as(), or defaults
// to stub_name via operator().
//
// Usage:
//   // header:
//   using mm_fn = void(*)(const Tensor&, const Tensor&, Tensor&);
//   FLAGOS_DECLARE_DISPATCH(mm_fn, mm_stub)
//
//   // cpp:
//   FLAGOS_DEFINE_DISPATCH(mm_fn, mm_stub, "mm")
//   FLAGOS_REGISTER_DISPATCH(mm_fn, mm_stub, FlagosDevice::FlagOS, my_flagos_mm)
//   FLAGOS_REGISTER_DISPATCH(mm_fn, mm_stub, FlagosDevice::CUDA,   my_cuda_mm)
//
//   // call (uses stub_name "mm" for config lookup):
//   mm_stub(self, mat2, out);
//
//   // call with op name override (uses "mm.out" for config lookup):
//   mm_stub.dispatch_as("mm.out", self, mat2, out);

template <typename FnPtr>
class FlagosDispatchStub {
 public:
  explicit FlagosDispatchStub(const char* stub_name) : stub_name_(stub_name) {}

  void register_kernel(FlagosDevice device, FnPtr fn) {
    switch (device) {
      case FlagosDevice::CUDA:   cuda_fn_ = fn;   break;
      case FlagosDevice::FlagOS: flagos_fn_ = fn;  break;
      case FlagosDevice::NPU:    npu_fn_ = fn;     break;
      case FlagosDevice::MUSA:   musa_fn_ = fn;    break;
    }
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    return dispatch_as(stub_name_, std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto dispatch_as(const std::string& op_name, Args&&... args) const {
    auto backend = get_backend_for_op(op_name);
    log_dispatch(op_name, backend);
    auto fn = get_fn(backend);
    TORCH_CHECK(fn, op_name, ": backend not registered");
    return fn(std::forward<Args>(args)...);
  }

 private:
  FnPtr get_fn(FlagosDevice device) const {
    switch (device) {
      case FlagosDevice::CUDA:   return cuda_fn_;
      case FlagosDevice::FlagOS: return flagos_fn_;
      case FlagosDevice::NPU:    return npu_fn_;
      case FlagosDevice::MUSA:   return musa_fn_;
    }
    return nullptr;
  }

  static void log_dispatch(const std::string& op_name, FlagosDevice backend) {
    static const bool enabled = []() {
      const char* v = std::getenv("FLAGOS_LOG_DISPATCH");
      return v && std::string(v) == "1";
    }();
    if (!enabled) return;
    const char* name;
    switch (backend) {
      case FlagosDevice::CUDA:   name = "cuda"; break;
      case FlagosDevice::FlagOS: name = "flaggems"; break;
      case FlagosDevice::NPU:    name = "npu"; break;
      case FlagosDevice::MUSA:   name = "musa"; break;
      default:                   name = "unknown"; break;
    }
    fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(), name);
  }

  const char* stub_name_;
  FnPtr cuda_fn_   = nullptr;
  FnPtr flagos_fn_ = nullptr;
  FnPtr npu_fn_    = nullptr;
  FnPtr musa_fn_   = nullptr;
};

namespace detail {
template <typename FnPtr>
struct DispatchRegistrar {
  DispatchRegistrar(FlagosDispatchStub<FnPtr>& stub, FlagosDevice device, FnPtr fn) {
    stub.register_kernel(device, fn);
  }
};
} // namespace detail

} // namespace at::native::flagos

#define FLAGOS_DECLARE_DISPATCH(fn_type, name) \
  extern ::at::native::flagos::FlagosDispatchStub<fn_type> name;

#define FLAGOS_DEFINE_DISPATCH(fn_type, name, stub_name) \
  ::at::native::flagos::FlagosDispatchStub<fn_type> name(stub_name);

#define FLAGOS_REGISTER_DISPATCH_UID2(fn_type, name, device, fn, uid) \
  static ::at::native::flagos::detail::DispatchRegistrar<fn_type>    \
      name##_registrar_##uid(name, device, fn);
#define FLAGOS_REGISTER_DISPATCH_UID(fn_type, name, device, fn, uid) \
  FLAGOS_REGISTER_DISPATCH_UID2(fn_type, name, device, fn, uid)
#define FLAGOS_REGISTER_DISPATCH(fn_type, name, device, fn) \
  FLAGOS_REGISTER_DISPATCH_UID(fn_type, name, device, fn, __COUNTER__)
