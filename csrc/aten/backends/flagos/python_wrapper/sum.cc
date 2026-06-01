// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../sum.h"

namespace at::native::flagos {

namespace {

at::Tensor SumDimKernelPython(const at::Tensor& self, at::OptionalIntArrayRef dim,
                              bool keepdim, std::optional<at::ScalarType> dtype) {
  // FlagGems sum_dim_comm has a recursion bug: when dim is None it calls
  // torch.sum(inp, dtype=dtype) which re-dispatches to sum_dim → infinite loop.
  // Normalize None to [] so FlagGems takes the dim==[] path (calls local sum()).
  if (!dim.has_value()) {
    std::vector<int64_t> empty_dim;
    return CallPythonOp_TOIB("sum_dim", self,
                             at::OptionalIntArrayRef(empty_dim),
                             keepdim, dtype);
  }
  return CallPythonOp_TOIB("sum_dim", self, dim, keepdim, dtype);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SumDimFn, sum_dim_dispatcher, Backend::kFlagOsPython, SumDimKernelPython)

} // namespace at::native::flagos
