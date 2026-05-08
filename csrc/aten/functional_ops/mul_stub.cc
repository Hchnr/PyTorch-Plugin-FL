// Copyright (c) 2026, BAAI. All rights reserved.

#include "mul_stub.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/mul_meta.h>
#include <ATen/ops/mul_native.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(MulTensorFn, mul_tensor_stub, "mul.Tensor")

namespace {

at::Tensor mul_kernel_cuda(
    const at::Tensor& self, const at::Tensor& other) {
  struct cuda_impl final : public at::native::structured_mul_out {
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
  op.meta(self, other);
  op.impl(self, other, op.out_);
  return op.out_;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MulTensorFn, mul_tensor_stub, FlagosDevice::kCuda, mul_kernel_cuda)

} // namespace at::native::flagos
