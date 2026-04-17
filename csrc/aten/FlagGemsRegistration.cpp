// Copyright (c) 2026, BAAI. All rights reserved.
//
// FlagGems operator registration via C++ wrapper for high-performance dispatch.
// This replaces Python-level registration (torch.library.Library) with direct
// C++ TORCH_LIBRARY_IMPL, reducing CPU overhead significantly.

#include <torch/library.h>

#ifdef FLAGGEMS_AVAILABLE

#include "flag_gems/operators.h"

namespace at::flagos {

// Register FlagGems operators for PrivateUse1 (flagos) dispatch key
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // BLAS operations
  m.impl("addmm", TORCH_FN(flag_gems::addmm));
  m.impl("mm", TORCH_FN(flag_gems::mm_tensor));
  m.impl("bmm", TORCH_FN(flag_gems::bmm));

  // Reduction operations
  m.impl("sum", TORCH_FN(flag_gems::sum));
  m.impl("sum.dim_IntList", TORCH_FN(flag_gems::sum_dim));
  m.impl("max", TORCH_FN(flag_gems::max));
  m.impl("max.dim", TORCH_FN(flag_gems::max_dim));
  m.impl("max.dim_max", TORCH_FN(flag_gems::max_dim_max));

  // Normalization
  m.impl("rms_norm", TORCH_FN(flag_gems::rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(flag_gems::fused_add_rms_norm));

  // Tensor operations
  m.impl("cat", TORCH_FN(flag_gems::cat));
  m.impl("embedding", TORCH_FN(flag_gems::embedding));
  m.impl("embedding_backward", TORCH_FN(flag_gems::embedding_backward));
  m.impl("argmax", TORCH_FN(flag_gems::argmax));
  m.impl("nonzero", TORCH_FN(flag_gems::nonzero));
  m.impl("topk", TORCH_FN(flag_gems::topk));

  // Division operations
  m.impl("div.Tensor", TORCH_FN(flag_gems::true_div));
  m.impl("div_.Tensor", TORCH_FN(flag_gems::true_div_));
  m.impl("div.Scalar", TORCH_FN(flag_gems::true_div));
  m.impl("div_.Scalar", TORCH_FN(flag_gems::true_div_));
  m.impl("div.Tensor_mode", TORCH_FN(flag_gems::div_mode));
  m.impl("div_.Tensor_mode", TORCH_FN(flag_gems::div_mode_));
  m.impl("div.Scalar_mode", TORCH_FN(flag_gems::div_mode));
  m.impl("div_.Scalar_mode", TORCH_FN(flag_gems::div_mode_));
  m.impl("floor_divide", TORCH_FN(flag_gems::floor_div));
  m.impl("floor_divide_.Tensor", TORCH_FN(flag_gems::floor_div_));
  m.impl("floor_divide.Scalar", TORCH_FN(flag_gems::floor_div));
  m.impl("floor_divide_.Scalar", TORCH_FN(flag_gems::floor_div_));
  m.impl("divide.Tensor", TORCH_FN(flag_gems::true_div));
  m.impl("divide_.Tensor", TORCH_FN(flag_gems::true_div_));
  m.impl("divide.Scalar", TORCH_FN(flag_gems::true_div));
  m.impl("divide_.Scalar", TORCH_FN(flag_gems::true_div_));
  m.impl("divide.Tensor_mode", TORCH_FN(flag_gems::div_mode));
  m.impl("divide_.Tensor_mode", TORCH_FN(flag_gems::div_mode_));
  m.impl("divide.Scalar_mode", TORCH_FN(flag_gems::div_mode));
  m.impl("divide_.Scalar_mode", TORCH_FN(flag_gems::div_mode_));
  m.impl("true_divide.Tensor", TORCH_FN(flag_gems::true_div));
  m.impl("true_divide_.Tensor", TORCH_FN(flag_gems::true_div_));
  m.impl("remainder.Scalar", TORCH_FN(flag_gems::remainder));
  m.impl("remainder_.Scalar", TORCH_FN(flag_gems::remainder_));
  m.impl("remainder.Tensor", TORCH_FN(flag_gems::remainder));
  m.impl("remainder_.Tensor", TORCH_FN(flag_gems::remainder_));
  m.impl("remainder.Scalar_Tensor", TORCH_FN(flag_gems::remainder));

  // Sorting
  m.impl("sort", TORCH_FN(flag_gems::sort));
  m.impl("sort.stable", TORCH_FN(flag_gems::sort_stable));

  // Fill operations
  m.impl("fill.Scalar", TORCH_FN(flag_gems::fill_scalar));
  m.impl("fill.Tensor", TORCH_FN(flag_gems::fill_tensor));
  m.impl("fill_.Scalar", TORCH_FN(flag_gems::fill_scalar_));
  m.impl("fill_.Tensor", TORCH_FN(flag_gems::fill_tensor_));

  // Softmax
  m.impl("softmax", TORCH_FN(flag_gems::softmax));
  m.impl("softmax_backward", TORCH_FN(flag_gems::softmax_backward));

  // Copy operations
  m.impl("to_copy", TORCH_FN(flag_gems::to_copy));
  m.impl("copy_", TORCH_FN(flag_gems::copy_));

  // Factory operations
  m.impl("zeros", TORCH_FN(flag_gems::zeros));
  m.impl("exponential_", TORCH_FN(flag_gems::exponential_));
}

// Register FlagGems custom operators
TORCH_LIBRARY_IMPL(flag_gems, PrivateUse1, m) {
  m.impl("rotary_embedding", TORCH_FN(flag_gems::rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(flag_gems::rotary_embedding_inplace));
  m.impl("reshape_and_cache_flash", TORCH_FN(flag_gems::reshape_and_cache_flash));
  m.impl("flash_attn_varlen_func", TORCH_FN(flag_gems::flash_attn_varlen_func));
  m.impl("rwkv_mm_sparsity", TORCH_FN(flag_gems::rwkv_mm_sparsity));
  m.impl("rwkv_ka_fusion", TORCH_FN(flag_gems::rwkv_ka_fusion));
}

} // namespace at::flagos

#endif // FLAGGEMS_AVAILABLE
