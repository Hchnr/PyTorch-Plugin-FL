// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../DispatchStub.h"

namespace at::native::flagos {

using bmm_fn = void (*)(const at::Tensor&, const at::Tensor&, at::Tensor&);
FLAGOS_DECLARE_DISPATCH(bmm_fn, bmm_stub)

} // namespace at::native::flagos
