// Copyright (c) 2026, BAAI. All rights reserved.

#include "rsqrt_stub.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/rsqrt_meta.h>
#include <ATen/ops/rsqrt_native.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(RsqrtFn, rsqrt_stub, "rsqrt")

namespace {

at::Tensor rsqrt_kernel_cuda(const at::Tensor& self) {
  struct cuda_impl final : public at::native::structured_rsqrt_out {
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

FLAGOS_REGISTER_DISPATCH(RsqrtFn, rsqrt_stub, FlagosDevice::kCuda, rsqrt_kernel_cuda)

} // namespace at::native::flagos
