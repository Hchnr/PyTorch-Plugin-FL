// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "Common.h"
#include <ATen/ops/mm_meta.h>
#include <ATen/ops/mm_native.h>

namespace at::native::flagos {

// Unified structured mm op for PrivateUse1 (flagos) dispatch key.
// meta() is inherited from at::meta::structured_mm (PyTorch-generated, not hand-written).
// impl() dispatches to the physical backend selected by flagos_device_.
struct structured_mm_out_flagos final : public at::meta::structured_mm {
  explicit structured_mm_out_flagos(at::Tensor& out, FlagosDevice device)
      : out_(out), flagos_device_(device) {}

  void set_output_strided(
      int64_t output_idx,
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      at::TensorOptions options,
      at::DimnameList names) override;

  void set_output_raw_strided(
      int64_t output_idx,
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      at::TensorOptions options,
      at::DimnameList names) override;

  const at::Tensor& maybe_get_output(int64_t output_idx) override;

  // Dispatch impl to the selected backend.
  void impl(const at::Tensor& self, const at::Tensor& mat2);

  at::Tensor& out_;
  FlagosDevice flagos_device_;
};

} // namespace at::native::flagos
