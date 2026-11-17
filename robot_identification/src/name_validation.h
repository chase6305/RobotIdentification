#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace robot_identification::detail {

inline bool HasUniqueNonemptyNames(const std::vector<std::string>& names) {
    if (names.empty()) return false;
    std::vector<std::string_view> ordered(names.begin(), names.end());
    std::sort(ordered.begin(), ordered.end());
    return !ordered.front().empty() && std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end();
}

}  // namespace robot_identification::detail
