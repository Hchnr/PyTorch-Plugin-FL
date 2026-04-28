// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../DispatchStub.h"

namespace at::native::flagos {

using cat_fn = at::Tensor (*)(const at::ITensorListRef&, int64_t);
FLAGOS_DECLARE_DISPATCH(cat_fn, cat_stub)

} // namespace at::native::flagos
