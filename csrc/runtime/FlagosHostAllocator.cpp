// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <c10/core/Allocator.h>
#include <ATen/core/TensorBase.h>

#include <accelerator/include/flagos.h>

namespace c10::flagos {

namespace {

struct FlagosHostAllocator final : c10::Allocator {
  FlagosHostAllocator() = default;

  static void ReportAndDelete(void* ptr) {
    if (ptr) {
      foFreeHost(ptr);
    }
  }

  c10::DataPtr allocate(size_t size) override {
    void* ptr = nullptr;
    if (size > 0) {
      foMallocHost(&ptr, size);
    }
    return {ptr, ptr, &ReportAndDelete, c10::DeviceType::CPU};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    foMemcpy(dest, src, count, foMemcpyHostToHost);
  }
};

static FlagosHostAllocator flagos_host_alloc;

} // namespace

c10::Allocator* getFlagosHostAllocator() {
  return &flagos_host_alloc;
}

} // namespace c10::flagos
