#pragma once

#include <yaml-cpp/yaml.h>

namespace util_yaml {

template <typename T>
inline T as(const YAML::Node& node, const T& fallback) {
    if (!node.IsDefined()) {
        return fallback;
    }
    return node.as<T>();
}
} // namespace util_yaml