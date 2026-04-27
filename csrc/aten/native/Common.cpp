// Copyright (c) 2026, BAAI. All rights reserved.

#include "Common.h"

#include <cstdio>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace at::native::flagos {

namespace {

std::string default_config_path() {
  // Resolve relative to this shared library's location at runtime would require
  // dladdr; instead use the source-tree convention: the config lives at
  // <package_root>/torch_fl/backends.conf, where package_root is two directories
  // above this file's build output.  Users can always override via
  // FLAGOS_BACKEND_CONFIG.
  return FLAGOS_SOURCE_ROOT "/torch_fl/backends.conf";
}

std::unordered_map<std::string, FlagosDevice> load_backend_config() {
  std::unordered_map<std::string, FlagosDevice> table;

  const char* env = std::getenv("FLAGOS_BACKEND_CONFIG");
  std::string path = env ? env : default_config_path();

  std::ifstream f(path);
  if (!f.is_open()) {
    return table;
  }

  fprintf(stderr, "[flagos] loading backend config from %s\n", path.c_str());

  std::string line;
  while (std::getline(f, line)) {
    // strip comments
    auto comment = line.find('#');
    if (comment != std::string::npos) line = line.substr(0, comment);

    auto eq = line.find('=');
    if (eq == std::string::npos) continue;

    auto trim = [](std::string s) {
      size_t l = s.find_first_not_of(" \t\r\n");
      size_t r = s.find_last_not_of(" \t\r\n");
      return (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
    };

    std::string op = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));

    if (op.empty() || val.empty()) continue;

    if (val == "cuda") {
      table[op] = FlagosDevice::CUDA;
    } else if (val == "flagos" || val == "flaggems") {
      table[op] = FlagosDevice::FlagOS;
    } else {
      fprintf(stderr, "[flagos] unknown backend '%s' for op '%s', using flagos\n", val.c_str(), op.c_str());
      table[op] = FlagosDevice::FlagOS;
    }
  }

  // Per-op env var overrides: FLAGOS_OP_<op_name>=cuda|flaggems
  // e.g. FLAGOS_OP_mm=cuda  or  FLAGOS_OP_mm__out=cuda
  // Dots in op names are replaced with double underscores to avoid ambiguity
  // with ops that already contain underscores (e.g. mm_out vs mm.out).
  for (auto& [op, _] : table) {
    std::string key = "FLAGOS_OP_";
    for (char c : op) {
      if (c == '.') key += "__";
      else key += c;
    }
    const char* override_val = std::getenv(key.c_str());
    if (!override_val) continue;
    std::string v(override_val);
    if (v == "cuda") {
      table[op] = FlagosDevice::CUDA;
      fprintf(stderr, "[flagos] env override: %s -> cuda\n", op.c_str());
    } else if (v == "flagos" || v == "flaggems") {
      table[op] = FlagosDevice::FlagOS;
      fprintf(stderr, "[flagos] env override: %s -> flaggems\n", op.c_str());
    }
  }

  return table;
}

const std::unordered_map<std::string, FlagosDevice>& backend_table() {
  static const auto table = load_backend_config();
  return table;
}

} // namespace

FlagosDevice get_backend_for_op(const std::string& op_name) {
  const auto& table = backend_table();
  auto it = table.find(op_name);
  return it != table.end() ? it->second : FlagosDevice::FlagOS;
}

void log_dispatch(const std::string& op_name, FlagosDevice backend) {
  static const bool enabled = []() {
    const char* v = std::getenv("FLAGOS_LOG_DISPATCH");
    return v && std::string(v) == "1";
  }();
  if (!enabled) return;
  const char* name = (backend == FlagosDevice::CUDA) ? "cuda" : "flaggems";
  fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(), name);
}

} // namespace at::native::flagos
