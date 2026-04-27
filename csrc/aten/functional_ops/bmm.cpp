// Copyright (c) 2026, BAAI. All rights reserved.

#include "bmm.h"
#include "bmm_stub.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>

namespace at::native::flagos {

void structured_bmm_out_flagos::set_output_strided(
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

void structured_bmm_out_flagos::set_output_raw_strided(
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

const at::Tensor& structured_bmm_out_flagos::maybe_get_output(int64_t) {
  return out_;
}

void structured_bmm_out_flagos::impl(const at::Tensor& self, const at::Tensor& mat2, const std::string& op_name) {
  bmm_stub.dispatch_as(op_name, self, mat2, out_);
}

} // namespace at::native::flagos
