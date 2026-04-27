// Copyright (c) 2026, BAAI. All rights reserved.

#include "mm.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/mm_native.h>
#include <c10/util/Exception.h>
#include <flag_gems/operators.h>

namespace at::native::flagos {

void structured_mm_out_flagos::set_output_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

void structured_mm_out_flagos::set_output_raw_strided(
    int64_t output_idx,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::TensorOptions options,
    at::DimnameList names) {
  at::native::resize_output(out_, sizes);
  if (!names.empty()) {
    at::namedinference::propagate_names(out_, names);
  }
}

const at::Tensor& structured_mm_out_flagos::maybe_get_output(int64_t) {
  return out_;
}

void structured_mm_out_flagos::impl(const at::Tensor& self, const at::Tensor& mat2) {
  switch (flagos_device_) {
    case FlagosDevice::CUDA: {
      struct cuda_impl final : public at::native::structured_mm_out_cuda {
        explicit cuda_impl(at::Tensor& out) : out_(out) {}
        void set_output_raw_strided(
            int64_t, at::IntArrayRef sizes, at::IntArrayRef,
            at::TensorOptions, at::DimnameList) override {
          at::native::resize_output(out_, sizes);
        }
        const at::Tensor& maybe_get_output(int64_t) override { return out_; }
        at::Tensor& out_;
      };
      cuda_impl op(out_);
      op.impl(self, mat2, out_);
      break;
    }
    case FlagosDevice::FlagOS:
      flag_gems::mm_out_tensor(self, mat2, out_);
      break;
    case FlagosDevice::NPU:
    case FlagosDevice::MUSA:
      TORCH_CHECK(false, "mm: NPU/MUSA backend not yet implemented in flagos wrapper");
      break;
  }
}

} // namespace at::native::flagos
