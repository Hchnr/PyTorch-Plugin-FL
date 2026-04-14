// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.h
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/core/CachingHostAllocator.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <accelerator/include/flagos.h>

namespace c10::flagos {

struct FlagosDeviceAllocator final : at::Allocator {
  FlagosDeviceAllocator() = default;

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    foFree(ptr);
  }

  at::DataPtr allocate(size_t nbytes) override {
    int current_device_index = -1;
    foGetDevice(&current_device_index);

    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
    void* data = nullptr;
    if (nbytes > 0) {
      foMalloc(&data, nbytes);
      TORCH_CHECK(
          data, "Failed to allocate ", nbytes, " bytes on flagos device.");
    }
    return {data, data, &ReportAndDelete, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    foMemcpy(dest, src, count, foMemcpyDeviceToDevice);
  }
};

} // namespace c10::flagos
