// Copyright (c) 2026, BAAI. All rights reserved.

#include "cat_stub.h"

#include <ATen/ops/cat_cuda_dispatch.h>
#include <flag_gems/operators.h>

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(cat_fn, cat_stub, "cat")

namespace {

at::Tensor cat_kernel_flaggems(const at::ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  std::vector<at::Tensor> tensor_vec(materialized.begin(), materialized.end());
  return flag_gems::cat(tensor_vec, dim);
}

at::Tensor cat_kernel_cuda(const at::ITensorListRef& tensors, int64_t dim) {
  return at::cuda::cat(tensors, dim);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(cat_fn, cat_stub, FlagosDevice::FlagOS, cat_kernel_flaggems)
FLAGOS_REGISTER_DISPATCH(cat_fn, cat_stub, FlagosDevice::CUDA,   cat_kernel_cuda)

} // namespace at::native::flagos
