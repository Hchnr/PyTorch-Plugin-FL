// Copyright (c) 2026, BAAI. All rights reserved.

#include "add_stub.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/add_meta.h>
#include <ATen/ops/add_native.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(AddTensorFn, add_tensor_stub, "add.Tensor")

namespace {

at::Tensor add_kernel_cuda(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  struct cuda_impl final : public at::native::structured_ufunc_add_CUDA {
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
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.out_);
  return op.out_;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AddTensorFn, add_tensor_stub, FlagosDevice::kCuda, add_kernel_cuda)

} // namespace at::native::flagos
