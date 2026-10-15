#pragma once
#include <cstdint>
#include <cstring>
#include <limits>

namespace slam {

inline bool finiteDouble(double value) noexcept {
    static_assert(sizeof(double)==sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

}  // namespace slam
