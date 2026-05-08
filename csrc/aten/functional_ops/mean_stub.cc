// Copyright (c) 2026, BAAI. All rights reserved.

#include "mean_stub.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/mean_meta.h>
#include <ATen/ops/mean_native.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(MeanDimFn, mean_dim_stub, "mean.dim")

namespace {

at::Tensor mean_dim_kernel_cuda(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype) {
  struct cuda_impl final : public at::native::structured_mean_out {
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
  op.meta(self, dim, keepdim, dtype);
  op.impl(self, dim, keepdim, dtype, op.out_);
  return op.out_;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MeanDimFn, mean_dim_stub, FlagosDevice::kCuda, mean_dim_kernel_cuda)

} // namespace at::native::flagos
