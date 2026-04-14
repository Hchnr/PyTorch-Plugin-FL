// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
// with namespace c10::openreg renamed to c10::flagos, OpenRegGeneratorImpl renamed to FlagosGeneratorImpl,
// and getDefaultOpenRegGenerator renamed to getDefaultFlagosGenerator.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "FlagosGenerator.h"

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

namespace c10::flagos {

const at::Generator& getDefaultFlagosGenerator(c10::DeviceIndex device_index) {
  static bool flag [[maybe_unused]] = []() {
    auto device_nums = device_count();
    default_generators.resize(device_nums);
    for (auto i = 0; i < device_nums; i++) {
      default_generators[i] = at::make_generator<FlagosGeneratorImpl>(i);
      default_generators[i].seed();
    }
    return true;
  }();

  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < device_count());
  }
  return default_generators[idx];
}

} // namespace c10::flagos
