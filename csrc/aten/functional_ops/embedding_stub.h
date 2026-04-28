// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../DispatchStub.h"

namespace at::native::flagos {

using embedding_fn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                                     int64_t, bool, bool);
FLAGOS_DECLARE_DISPATCH(embedding_fn, embedding_stub)

} // namespace at::native::flagos
