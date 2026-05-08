// Copyright (c) 2026, BAAI. All rights reserved.

#include "silu_stub.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/silu_meta.h>
#include <ATen/ops/silu_native.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(SiluFn, silu_stub, "silu")

namespace {

at::Tensor silu_kernel_cuda(const at::Tensor& self) {
  struct cuda_impl final : public at::native::structured_silu_out {
    at::Tensor out_;
    void set_output_strided(
        int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
        at::TensorOptions options, at::DimnameList) override {
      out_ = at::empty(sizes, options);
    }
    void set_output_raw_strided(
        int64_t, at::IntArrayRef sizes, at::IntArrayRef strides,
        at::TensorOptions options, at::DimnameList) override {
      out_ = at::empty(sizes, options);
    }
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
  };

  cuda_impl op;
  op.meta(self);
  op.impl(self, op.out_);
  return op.out_;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SiluFn, silu_stub, FlagosDevice::kCuda, silu_kernel_cuda)

} // namespace at::native::flagos
