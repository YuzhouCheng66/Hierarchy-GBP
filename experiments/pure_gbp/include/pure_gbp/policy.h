#pragma once
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace pure_gbp {
inline int policyInteger(const char* key,int fallback,int minimum=1) {
    const char* value=std::getenv(key);
    if(!value)return fallback;
    char* end=nullptr;errno=0;
    const long parsed=std::strtol(value,&end,10);
    if(errno || end==value || *end || parsed<minimum || parsed>1000000)
        throw std::runtime_error(std::string("invalid pure GBP policy: ")+key);
    return int(parsed);
}
inline double policyTolerance(const char* key,double fallback) {
    const char* value=std::getenv(key);
    if(!value)return fallback;
    char* end=nullptr;errno=0;
    const double parsed=std::strtod(value,&end);
    if(errno || end==value || *end || !std::isfinite(parsed) || !(parsed>0 && parsed<1))
        throw std::runtime_error(std::string("invalid pure GBP tolerance: ")+key);
    return parsed;
}
}
