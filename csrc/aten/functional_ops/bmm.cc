// Copyright (c) 2026, BAAI. All rights reserved.

#include "bmm.h"
#include "bmm_stub.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>

namespace at::native::flagos {

void StructuredBmmOutFlagos::set_output_strided(
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

void StructuredBmmOutFlagos::set_output_raw_strided(
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

const at::Tensor& StructuredBmmOutFlagos::maybe_get_output(int64_t) {
  return out_;
}

void StructuredBmmOutFlagos::impl(const at::Tensor& self, const at::Tensor& mat2, const std::string& op_name) {
  bmm_stub.DispatchAs(op_name, self, mat2, out_);
}

} // namespace at::native::flagos
